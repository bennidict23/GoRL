# Third-party notice for `flow_policy`

This compatibility package is derived from the `playground/src/flow_policy`
package in [akanazawa/fpo](https://github.com/akanazawa/fpo), upstream commit
`418c2554f7cd22d52e14c07d951280929d73bf2f`.

`networks.py` and `ppo.py` add gradient clipping, a policy-scale cap, latent
regularization, and related diagnostics. Additional changes keep rollout-time
observation normalization fixed during PPO updates and correctly accumulate
running variance in `math_utils.py`. Explicit compatibility settings can replay
either archived observation-statistics behavior without changing the default
implementation. `rollouts.py` was modified to make W&B an optional, lazy
import. Modified files carry a notice at the top. The checked-in sources are
the public record of these modifications.

The upstream code is licensed under Apache-2.0. The complete license text is
included in the adjacent `LICENSE` file.
