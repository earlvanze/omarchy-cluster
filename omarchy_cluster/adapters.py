"""Model adapters that the omarchy-mlx checkout carries for mlx-lm.

The published mlx-community Muse Spark Open checkpoints (3-bit, 4-bit) declare model_type `muse_spark` in the Hugging Face text_config layout.
mlx-lm main has no loader for that name. omarchy-mlx ships one in `serve/mlx_omarchy_muse` (the class of mlx-lm pull request 1975, which uses only
mx operations, so it runs on Metal and on the Vulkan backend alike). `OMARCHY_MLX_SERVE` names that `serve` directory; every rank that loads a
model calls `register_adapters()` first, so the class exists before mlx-lm looks the model type up. Nothing happens when the package is absent.
"""
import os
import sys


def register_adapters():
    """Make `mlx_lm.models.muse_spark` resolvable. Returns True when an adapter was added, False when none was needed or found."""
    serve = os.environ.get("OMARCHY_MLX_SERVE")
    if serve and serve not in sys.path:
        sys.path.insert(0, serve)
    try:
        import mlx_omarchy_muse
    except ModuleNotFoundError as e:
        if e.name != "mlx_omarchy_muse":  # the package is there but one of its own imports is missing: say so
            raise
        return False
    return bool(mlx_omarchy_muse.register())
