# Third-party notices

GoRL builds on the following projects. Their licenses continue to apply to
their code.

## Flow Matching Policy Gradients (FPO)

The compatibility package in `src/flow_policy` is derived from
[akanazawa/fpo](https://github.com/akanazawa/fpo), upstream commit
`418c2554f7cd22d52e14c07d951280929d73bf2f`. The checked-in derivative adds
GoRL's gradient, policy-scale, latent-regularization, and observation-statistics
controls. The original `playground` code is licensed under Apache-2.0; a copy
of that license is kept next to the derived package.

The six non-Humanoid PPO cells and all FPO benchmark runs use the unmodified
upstream commit. Humanoid PPO uses the installed Brax and MuJoCo Playground
packages and has no FPO checkout dependency. DPPO runs apply only the
scalar-sigma likelihood fix from commit
`964dd78c6fb64de8c52eeb7ce80561c43455128f`. The modified source file carries
a `Modified by GoRL Authors` notice. Every run manifest records the selected
source commit and patch.

## Brax and MuJoCo Playground

The Humanoid profile uses
[Brax](https://github.com/google/brax) and
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) as
installed dependencies. Both projects are distributed under Apache-2.0.

## Weights & Biases

Optional experiment tracking uses the W&B Python client. GoRL does not
initialize a W&B run when tracking is disabled, and no account is required for
local runs. The pinned upstream baseline imports the client at module load, so
the training extra installs it even when external tracking is disabled.
