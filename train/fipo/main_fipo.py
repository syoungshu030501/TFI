"""
FIPO entry point — wraps verl.trainer.main_ppo, but first registers our
project-specific patches into verl's registries:

- train.fipo.verl_patches.future_kl_loss   -> POLICY_LOSS_REGISTRY["future_kl"]
- train.fipo.verl_patches.reward_manager   -> REWARD_MANAGER_REGISTRY["tfi_audit_v1"]

Run via train/fipo/launch.sh, do not invoke directly (Hydra config path resolution
needs verl on PYTHONPATH).
"""
import os
import sys

# Side-effect imports MUST happen before main() so that @register_policy_loss
# and @register decorators populate the verl registries (driver process).
# Ray spawn workers inherit the driver's PYTHONPATH (set by launch.sh),
# and Python's site machinery auto-imports sitecustomize.py from project root,
# which re-imports future_kl_loss and registers it in every worker process.
import train.fipo.verl_patches.future_kl_loss  # noqa: F401
import train.fipo.verl_patches.reward_manager  # noqa: F401

# Default to HF mirror for the bge encoder download (idempotent).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
