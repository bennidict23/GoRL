"""Standalone Brax PPO implementation for the Humanoid profile."""

from __future__ import annotations

import functools
import importlib.util
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .environment import BoundedLatentActionEnvironment, FrozenDecoder
from .types import (
    DecoderKind,
    EventSource,
    HumanoidBackendAvailability,
    HumanoidEvaluationConfig,
    HumanoidEvaluationResult,
    HumanoidLatentStageConfig,
    HumanoidMetricEvent,
    HumanoidNonFiniteMetricError,
    HumanoidPPOConfig,
    HumanoidPPOExecutionProfile,
    HumanoidPPOState,
    HumanoidTrainingResult,
    HumanoidWarmStart,
    Initialization,
    MetricCallback,
)


_REQUIRED_MODULES = ("brax", "jax", "mujoco_playground", "numpy")


@dataclass(frozen=True, slots=True)
class _Runtime:
    jax: Any
    jnp: Any
    np: Any
    brax_ppo: Any
    ppo_networks: Any
    running_statistics: Any
    registry: Any
    playground_wrapper: Any
    dm_control_suite: Any
    locomotion: Any
    manipulation: Any
    dm_control_suite_params: Any
    locomotion_params: Any
    manipulation_params: Any


def backend_availability() -> HumanoidBackendAvailability:
    """Check optional training dependencies without importing them."""

    missing = tuple(
        name for name in _REQUIRED_MODULES if importlib.util.find_spec(name) is None
    )
    if missing:
        return HumanoidBackendAvailability(
            available=False,
            missing_modules=missing,
            detail=(
                "Humanoid profile is unavailable; install the `train` extra. "
                f"Missing: {', '.join(missing)}"
            ),
        )
    return HumanoidBackendAvailability(
        available=True,
        missing_modules=(),
        detail="Brax and MuJoCo Playground modules are discoverable",
    )


def train_teacher(
    config: HumanoidPPOConfig,
    *,
    warm_start: HumanoidWarmStart | None = None,
    callback: MetricCallback | None = None,
    report_config: HumanoidEvaluationConfig | None = None,
    checkpoint_dir: str | Path | None = None,
) -> HumanoidTrainingResult:
    """Train the official PPO teacher.

    Omitting ``warm_start`` always means a fresh, randomly initialized policy,
    value function, and observation normalizer. The returned result records
    that fact as ``initialization="from_scratch"``.
    """

    runtime = _load_runtime()
    environment_config = runtime.registry.get_default_config(config.task)
    environment = runtime.registry.load(config.task, config=environment_config)
    eval_environment = None
    if config.execution_profile is HumanoidPPOExecutionProfile.OFFICIAL_DIRECT_COMPAT:
        eval_environment = runtime.registry.load(
            config.task,
            config=environment_config,
        )
    return _train(
        runtime=runtime,
        environment=environment,
        config=config,
        decoder_kind=DecoderKind.IDENTITY,
        action_size=int(environment.action_size),
        latent_size=int(environment.action_size),
        warm_start=warm_start,
        callback=callback,
        report_config=report_config,
        report_decoder=None,
        eval_environment=eval_environment,
        checkpoint_dir=checkpoint_dir,
    )


def train_latent_encoder(
    config: HumanoidLatentStageConfig,
    decoder: FrozenDecoder,
    *,
    warm_start: HumanoidWarmStart | None = None,
    callback: MetricCallback | None = None,
    report_config: HumanoidEvaluationConfig | None = None,
    checkpoint_dir: str | Path | None = None,
) -> HumanoidTrainingResult:
    """Train a bounded latent PPO policy while keeping ``decoder`` frozen."""

    runtime = _load_runtime()
    base_environment = _load_environment(runtime, config.ppo.task)
    environment = BoundedLatentActionEnvironment(
        base_environment,
        decoder,
        latent_size=config.latent_size,
        record_decoded_actions=config.record_decoded_actions,
    )
    return _train(
        runtime=runtime,
        environment=environment,
        config=config.ppo,
        decoder_kind=config.decoder_kind,
        action_size=int(base_environment.action_size),
        latent_size=int(environment.action_size),
        warm_start=warm_start,
        callback=callback,
        report_config=report_config,
        report_decoder=decoder,
        eval_environment=None,
        checkpoint_dir=checkpoint_dir,
    )


def evaluate_report(
    state: HumanoidPPOState,
    config: HumanoidEvaluationConfig = HumanoidEvaluationConfig(),
    *,
    decoder: FrozenDecoder | None = None,
    callback: MetricCallback | None = None,
) -> HumanoidEvaluationResult:
    """Run the canonical deterministic post-training report evaluation."""

    runtime = _load_runtime()
    base_environment = _load_environment(runtime, state.task)
    if state.decoder_kind is DecoderKind.IDENTITY:
        if decoder is not None:
            raise ValueError("identity PPO state does not use a decoder")
        environment = base_environment
    else:
        if decoder is None:
            raise ValueError(
                f"{state.decoder_kind.value} PPO state requires its frozen decoder"
            )
        environment = BoundedLatentActionEnvironment(
            base_environment,
            decoder,
            latent_size=state.latent_size,
        )
    _validate_state_compatibility(
        state,
        task=state.task,
        action_size=int(base_environment.action_size),
        latent_size=int(environment.action_size),
    )

    def normalize(observation: Any, _: Any) -> Any:
        return observation

    if bool(state.ppo_parameters.get("normalize_observations", False)):
        normalize = runtime.running_statistics.normalize
    network_factory = _network_factory(
        runtime,
        state.ppo_parameters,
        bounded=True,
    )
    network = network_factory(
        environment.observation_size,
        environment.action_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = runtime.ppo_networks.make_inference_fn(
        network,
        compute_value=False,
    )
    policy = make_policy(state.brax_params, deterministic=True)
    episode_length = (
        config.episode_length
        if config.episode_length is not None
        else int(state.ppo_parameters.get("episode_length", 1000))
    )

    @runtime.jax.jit
    def rollout(prng: Any) -> tuple[Any, Any]:
        reset_keys = runtime.jax.random.split(prng, config.num_envs)
        environment_state = runtime.jax.vmap(environment.reset)(reset_keys)

        def step_fn(carry: tuple[Any, Any, Any], _: Any):
            state_value, step_prng, done = carry
            step_prng, action_prng = runtime.jax.random.split(step_prng)
            action, _ = policy(state_value.obs, action_prng)
            next_state = runtime.jax.vmap(environment.step)(state_value, action)
            valid = ~done
            new_done = done | (next_state.done > 0.5)
            return (next_state, step_prng, new_done), (
                next_state.reward,
                valid,
            )

        initial_done = runtime.jnp.zeros((config.num_envs,), dtype=bool)
        _, outputs = runtime.jax.lax.scan(
            step_fn,
            (environment_state, prng, initial_done),
            None,
            length=episode_length,
        )
        return outputs

    rewards, valid_mask = rollout(runtime.jax.random.PRNGKey(config.seed))
    episode_returns = runtime.jnp.sum(rewards * valid_mask, axis=0)
    episode_lengths = runtime.jnp.sum(valid_mask, axis=0)
    returns_host = runtime.np.asarray(
        runtime.jax.device_get(episode_returns),
        dtype=float,
    )
    lengths_host = runtime.np.asarray(
        runtime.jax.device_get(episode_lengths),
        dtype=int,
    )
    metrics = {
        "eval/episode_reward": float(returns_host.mean()),
        "eval/episode_reward_std": float(returns_host.std()),
        "eval/episode_reward_min": float(returns_host.min()),
        "eval/episode_reward_max": float(returns_host.max()),
        "eval/episode_length_mean": float(lengths_host.mean()),
        "eval/environment_steps": float(lengths_host.sum()),
    }
    _validate_finite_metrics(
        metrics,
        source=EventSource.DETERMINISTIC_REPORT,
        env_steps=state.actual_environment_steps,
    )
    event = HumanoidMetricEvent(
        source=EventSource.DETERMINISTIC_REPORT,
        task=state.task,
        seed=config.seed,
        stage_index=state.stage_index,
        evaluation_index=0,
        requested_environment_steps=state.requested_environment_steps,
        actual_environment_steps=state.actual_environment_steps,
        deterministic=True,
        metrics=metrics,
    )
    if callback is not None:
        callback(event)
    return HumanoidEvaluationResult(
        task=state.task,
        seed=config.seed,
        stage_index=state.stage_index,
        deterministic=True,
        episode_returns=tuple(float(value) for value in returns_host),
        episode_lengths=tuple(int(value) for value in lengths_host),
        event=event,
    )


def _record_official_teacher_progress(
    deferred_metrics: list[tuple[int, Mapping[str, Any]]],
    step: int,
    metrics: Mapping[str, Any],
) -> None:
    reward = metrics.get("eval/episode_reward")
    if reward is not None:
        try:
            print(f"{int(step)}: reward={float(reward):.3f}", flush=True)
        except (BrokenPipeError, ValueError):
            pass
    deferred_metrics.append((int(step), dict(metrics)))


def _train(
    *,
    runtime: _Runtime,
    environment: Any,
    config: HumanoidPPOConfig,
    decoder_kind: DecoderKind,
    action_size: int,
    latent_size: int,
    warm_start: HumanoidWarmStart | None,
    callback: MetricCallback | None,
    report_config: HumanoidEvaluationConfig | None,
    report_decoder: FrozenDecoder | None,
    eval_environment: Any | None,
    checkpoint_dir: str | Path | None,
) -> HumanoidTrainingResult:
    ppo_parameters = _official_ppo_parameters(runtime, config.task)
    ppo_parameters.update(config.ppo_overrides)
    ppo_parameters["num_timesteps"] = config.requested_environment_steps
    ppo_parameters["num_evals"] = config.num_evals
    compatible = (
        config.execution_profile is HumanoidPPOExecutionProfile.OFFICIAL_DIRECT_COMPAT
    )
    teacher_phase = decoder_kind is DecoderKind.IDENTITY
    network_factory = _network_factory(
        runtime,
        ppo_parameters,
        bounded=True,
        implicit_bounded_default=compatible,
    )
    training_parameters = dict(ppo_parameters)
    training_parameters.pop("network_factory", None)

    initialization = (
        Initialization.FROM_SCRATCH if warm_start is None else Initialization.WARM_START
    )
    restore_params = None
    restore_value_fn = True
    if warm_start is not None:
        _validate_state_compatibility(
            warm_start.state,
            task=config.task,
            action_size=action_size,
            latent_size=latent_size,
        )
        restore_params, restore_value_fn = _prepare_restore_params(
            runtime=runtime,
            environment=environment,
            config=config,
            training_parameters=training_parameters,
            network_factory=network_factory,
            warm_start=warm_start,
        )
    if compatible and teacher_phase and warm_start is not None:
        raise ValueError(
            "official_direct_compat teacher execution supports fresh training only"
        )

    events: list[HumanoidMetricEvent] = []
    # Brax reports the pre-update evaluation before its first
    # ``policy_params_fn`` callback. Native execution can associate that
    # evaluation with the warm-start checkpoint. Historical latent execution
    # did not: only parameters exposed by a later callback were checkpointable.
    latest: dict[str, Any] = {
        "step": 0,
        "params": (None if compatible and not teacher_phase else restore_params),
    }
    best: dict[str, Any] = {"reward": None, "params": None, "step": 0}
    deferred_metrics: list[tuple[int, Mapping[str, Any]]] = []
    resolved_checkpoint_dir = (
        None if checkpoint_dir is None else Path(checkpoint_dir).resolve()
    )
    if compatible and resolved_checkpoint_dir is None:
        raise ValueError(
            "official_direct_compat execution requires a private checkpoint directory"
        )
    if resolved_checkpoint_dir is not None:
        resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def policy_params_fn(step: int, make_policy: Any, params: Any) -> None:
        del make_policy
        latest["step"] = int(step)
        latest["params"] = params

    def progress_fn(step: int, metrics: Mapping[str, Any]) -> None:
        scalar_metrics = _scalar_metrics(
            runtime,
            metrics,
            source=EventSource.TRAINING_EVAL,
            env_steps=int(step),
        )
        event = HumanoidMetricEvent(
            source=EventSource.TRAINING_EVAL,
            task=config.task,
            seed=config.seed,
            stage_index=config.stage_index,
            evaluation_index=len(events),
            requested_environment_steps=config.requested_environment_steps,
            actual_environment_steps=int(step),
            deterministic=config.training_eval_deterministic,
            metrics=scalar_metrics,
        )
        events.append(event)
        if callback is not None:
            callback(event)

        reward = event.return_mean
        checkpoint_improved = best["reward"] is None or (
            reward is not None
            and (
                reward >= float(best["reward"]) - 1e-6
                if compatible and not teacher_phase
                else reward >= float(best["reward"])
            )
        )
        if reward is not None and latest["params"] is not None and checkpoint_improved:
            best["reward"] = reward
            # Training state is donated between Brax epochs. Snapshot improved
            # checkpoints onto the host while the callback parameters are valid.
            best["params"] = runtime.jax.device_get(latest["params"])
            best["step"] = int(step)

            if compatible and not teacher_phase and resolved_checkpoint_dir is not None:
                checkpoint_path = resolved_checkpoint_dir / "best_params.pkl"
                with checkpoint_path.open("wb") as stream:
                    pickle.dump(
                        best["params"],
                        stream,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

    def official_teacher_policy_params_fn(*_: Any) -> None:
        return None

    def official_teacher_progress_fn(
        step: int,
        metrics: Mapping[str, Any],
    ) -> None:
        _record_official_teacher_progress(deferred_metrics, step, metrics)

    if compatible:
        for key, default in (
            ("clipping_epsilon", 0.3),
            ("gae_lambda", 0.95),
            ("normalize_advantage", True),
        ):
            if training_parameters.get(key) == default:
                training_parameters.pop(key, None)

    selected_progress_fn = (
        official_teacher_progress_fn if compatible and teacher_phase else progress_fn
    )
    selected_policy_params_fn = (
        official_teacher_policy_params_fn
        if compatible and teacher_phase
        else policy_params_fn
    )
    train_kwargs: dict[str, Any] = {
        "environment": environment,
        "num_eval_envs": config.num_eval_envs,
        "seed": config.seed,
        "progress_fn": selected_progress_fn,
        "policy_params_fn": selected_policy_params_fn,
        "wrap_env_fn": runtime.playground_wrapper.wrap_for_brax_training,
        "network_factory": network_factory,
        **training_parameters,
    }
    if compatible and teacher_phase:
        train_kwargs.update(
            {
                "eval_env": eval_environment,
                "restore_checkpoint_path": None,
                "save_checkpoint_path": str(resolved_checkpoint_dir),
            }
        )
    else:
        train_kwargs.update(
            {
                "restore_params": restore_params,
                "restore_value_fn": restore_value_fn,
            }
        )
        if not compatible or config.training_eval_deterministic:
            train_kwargs["deterministic_eval"] = config.training_eval_deterministic

    _, final_params, final_metrics = runtime.brax_ppo.train(**train_kwargs)
    if compatible and teacher_phase:
        for step, metrics in deferred_metrics:
            scalar_metrics = _scalar_metrics(
                runtime,
                metrics,
                source=EventSource.TRAINING_EVAL,
                env_steps=step,
            )
            event = HumanoidMetricEvent(
                source=EventSource.TRAINING_EVAL,
                task=config.task,
                seed=config.seed,
                stage_index=config.stage_index,
                evaluation_index=len(events),
                requested_environment_steps=config.requested_environment_steps,
                actual_environment_steps=step,
                deterministic=config.training_eval_deterministic,
                metrics=scalar_metrics,
            )
            events.append(event)
            if callback is not None:
                callback(event)
        actual_steps = events[-1].actual_environment_steps if events else 0
        compatible_returns = [
            event for event in events if event.return_mean is not None
        ]
        if compatible_returns:
            selected_best = max(
                compatible_returns,
                key=lambda event: float(event.return_mean),
            )
            best["reward"] = selected_best.return_mean
            best["step"] = selected_best.actual_environment_steps
            if best["step"] == actual_steps:
                best["params"] = runtime.jax.device_get(final_params)
            elif best["step"] > 0 and resolved_checkpoint_dir is not None:
                checkpoint_path = resolved_checkpoint_dir / f"{best['step']:012d}"
                if checkpoint_path.exists():
                    from brax.training.agents.ppo import checkpoint as brax_checkpoint

                    best["params"] = runtime.jax.device_get(
                        brax_checkpoint.load(checkpoint_path)
                    )
    else:
        actual_steps = int(latest["step"])
    final_state = _state(
        config=config,
        decoder_kind=decoder_kind,
        action_size=action_size,
        latent_size=latent_size,
        actual_steps=actual_steps,
        ppo_parameters=ppo_parameters,
        params=final_params,
    )
    best_state = None
    if best["params"] is not None:
        best_state = _state(
            config=config,
            decoder_kind=decoder_kind,
            action_size=action_size,
            latent_size=latent_size,
            actual_steps=int(best["step"]),
            ppo_parameters=ppo_parameters,
            params=best["params"],
        )

    final_scalar_metrics = _scalar_metrics(
        runtime,
        final_metrics,
        source="training_final",
        env_steps=actual_steps,
    )
    final_return = final_scalar_metrics.get("eval/episode_reward")
    if final_return is None and events:
        final_return = events[-1].return_mean
    observed_returns = [
        event.return_mean for event in events if event.return_mean is not None
    ]
    if final_return is not None:
        observed_returns.append(final_return)
    best_return = max(observed_returns) if observed_returns else None
    if best["reward"] != best_return and not (compatible and not teacher_phase):
        # Brax invokes its initial progress callback before exposing the
        # corresponding parameters. Keep the scalar overall best, but do not
        # attach an unrelated checkpoint when that initial policy was best.
        best_state = None

    report = None
    if report_config is not None:
        report = evaluate_report(
            final_state,
            report_config,
            decoder=report_decoder,
            callback=callback,
        )
    return HumanoidTrainingResult(
        initialization=initialization,
        state=final_state,
        best_state=best_state,
        events=tuple(events),
        final_return=final_return,
        best_return=best_return,
        report=report,
    )


def _prepare_restore_params(
    *,
    runtime: _Runtime,
    environment: Any,
    config: HumanoidPPOConfig,
    training_parameters: Mapping[str, Any],
    network_factory: Any,
    warm_start: HumanoidWarmStart,
) -> tuple[tuple[Any, Any, Any], bool]:
    if warm_start.restore_policy and warm_start.restore_normalizer:
        return warm_start.state.brax_params, warm_start.restore_value

    initial_parameters = dict(training_parameters)
    initial_parameters["num_timesteps"] = 0
    initial_parameters["num_evals"] = 0
    _, fresh_params, _ = runtime.brax_ppo.train(
        environment=environment,
        num_eval_envs=config.num_eval_envs,
        seed=config.seed,
        wrap_env_fn=runtime.playground_wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        run_evals=False,
        **initial_parameters,
    )
    old = warm_start.state.brax_params
    return (
        old[0] if warm_start.restore_normalizer else fresh_params[0],
        old[1] if warm_start.restore_policy else fresh_params[1],
        old[2] if warm_start.restore_value else fresh_params[2],
    ), True


def _state(
    *,
    config: HumanoidPPOConfig,
    decoder_kind: DecoderKind,
    action_size: int,
    latent_size: int,
    actual_steps: int,
    ppo_parameters: Mapping[str, Any],
    params: tuple[Any, Any, Any],
) -> HumanoidPPOState:
    return HumanoidPPOState(
        task=config.task,
        seed=config.seed,
        stage_index=config.stage_index,
        decoder_kind=decoder_kind,
        action_size=action_size,
        latent_size=latent_size,
        requested_environment_steps=config.requested_environment_steps,
        actual_environment_steps=actual_steps,
        ppo_parameters=ppo_parameters,
        normalizer_params=params[0],
        policy_params=params[1],
        value_params=params[2],
        training_eval_deterministic=config.training_eval_deterministic,
    )


def _validate_state_compatibility(
    state: HumanoidPPOState,
    *,
    task: str,
    action_size: int,
    latent_size: int,
) -> None:
    if state.task != task:
        raise ValueError(f"warm-start task mismatch: {state.task} != {task}")
    if state.action_size != action_size:
        raise ValueError(
            "warm-start environment action size mismatch: "
            f"{state.action_size} != {action_size}"
        )
    if state.latent_size != latent_size:
        raise ValueError(
            f"warm-start latent size mismatch: {state.latent_size} != {latent_size}"
        )
    if state.action_semantics != "bounded":
        raise ValueError("Humanoid warm-start must use bounded action semantics")


def _official_ppo_parameters(runtime: _Runtime, task: str) -> dict[str, Any]:
    if task in runtime.manipulation.ALL_ENVS:
        config = runtime.manipulation_params.brax_ppo_config(task)
    elif task in runtime.locomotion.ALL_ENVS:
        config = runtime.locomotion_params.brax_ppo_config(task)
    elif task in runtime.dm_control_suite.ALL_ENVS:
        config = runtime.dm_control_suite_params.brax_ppo_config(task)
    else:
        raise ValueError(f"MuJoCo Playground has no official PPO config for {task}")
    return dict(config)


def _load_environment(runtime: _Runtime, task: str) -> Any:
    environment_config = runtime.registry.get_default_config(task)
    return runtime.registry.load(task, config=environment_config)


def _network_factory(
    runtime: _Runtime,
    ppo_parameters: Mapping[str, Any],
    *,
    bounded: bool,
    implicit_bounded_default: bool = False,
) -> Any:
    network_kwargs = dict(ppo_parameters.get("network_factory") or {})
    if not (bounded and implicit_bounded_default):
        network_kwargs["distribution_type"] = "tanh_normal" if bounded else "normal"
    return functools.partial(
        runtime.ppo_networks.make_ppo_networks,
        **network_kwargs,
    )


def _scalar_metrics(
    runtime: _Runtime,
    metrics: Mapping[str, Any],
    *,
    source: EventSource | str,
    env_steps: int,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in metrics.items():
        try:
            array = runtime.np.asarray(runtime.jax.device_get(value))
        except (TypeError, ValueError):
            continue
        if array.size != 1:
            continue
        scalar = float(array.reshape(()))
        if not math.isfinite(scalar):
            raise HumanoidNonFiniteMetricError(
                source=source,
                metric=str(key),
                value=scalar,
                env_steps=env_steps,
            )
        values[str(key)] = scalar
    return values


def _validate_finite_metrics(
    metrics: Mapping[str, float],
    *,
    source: EventSource | str,
    env_steps: int,
) -> None:
    for key, value in metrics.items():
        scalar = float(value)
        if not math.isfinite(scalar):
            raise HumanoidNonFiniteMetricError(
                source=source,
                metric=str(key),
                value=scalar,
                env_steps=env_steps,
            )


def _load_runtime() -> _Runtime:
    availability = backend_availability()
    if not availability.available:
        raise ImportError(availability.detail)

    import jax
    import numpy as np
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as brax_ppo
    from jax import numpy as jnp
    from mujoco_playground import (
        dm_control_suite,
        locomotion,
        manipulation,
        registry,
        wrapper,
    )
    from mujoco_playground.config import (
        dm_control_suite_params,
        locomotion_params,
        manipulation_params,
    )

    return _Runtime(
        jax=jax,
        jnp=jnp,
        np=np,
        brax_ppo=brax_ppo,
        ppo_networks=ppo_networks,
        running_statistics=running_statistics,
        registry=registry,
        playground_wrapper=wrapper,
        dm_control_suite=dm_control_suite,
        locomotion=locomotion,
        manipulation=manipulation,
        dm_control_suite_params=dm_control_suite_params,
        locomotion_params=locomotion_params,
        manipulation_params=manipulation_params,
    )
