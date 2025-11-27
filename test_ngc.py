#!/usr/bin/env python
import sys
import textwrap
from importlib.metadata import version as pkg_version, PackageNotFoundError


def main() -> int:
    print("=== NGC / PyTorch / Transformer Engine capability check ===")
    print()

    # --- PyTorch / CUDA info ---
    try:
        import torch
    except Exception as e:
        print("PyTorch: NOT IMPORTABLE")
        print(f"  Error: {e}")
        return 1

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:  {cuda_available}")

    if not cuda_available:
        print("No CUDA device visible. Are you running with `--gpus all`?")
        return 1

    device_index = 0
    props = torch.cuda.get_device_properties(device_index)
    device_name = props.name
    major, minor = props.major, props.minor

    print(f"CUDA device:     {device_name}")
    print(f"SM capability:   {major}.{minor}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print()

    # --- Transformer Engine presence ---
    te_import_ok = False
    te_version = "unknown"
    try:
        import transformer_engine.pytorch as te  # noqa: F401

        te_import_ok = True
        try:
            te_version = pkg_version("transformer_engine")
        except PackageNotFoundError:
            # Fallback if metadata is missing
            te_version = getattr(te, "__version__", "unknown")
        print(f"Transformer Engine: present (version: {te_version})")
    except Exception as e:
        print("Transformer Engine: NOT IMPORTABLE")
        print(f"  Error: {e}")
        print()
        print_summary(
            te_version="unknown",
            fp8_status="UNAVAILABLE",
            fp8_reason="transformer_engine not importable",
            nvfp4_status="UNAVAILABLE",
            nvfp4_reason="transformer_engine not importable",
        )
        return 0  # Repo can still run in plain PyTorch

    print()

    # --- FP8 via TE check ---
    fp8_status, fp8_reason = check_te_fp8(device_name, major, minor, te_version)

    # --- NVFP4 via TE check (skips noisy path on RTX Blackwell) ---
    nvfp4_status, nvfp4_reason = check_nvfp4(device_name, major, minor, te_version)

    print()
    print_summary(
        te_version=te_version,
        fp8_status=fp8_status,
        fp8_reason=fp8_reason,
        nvfp4_status=nvfp4_status,
        nvfp4_reason=nvfp4_reason,
    )
    return 0


def check_te_fp8(device_name: str, major: int, minor: int, te_version: str):
    """
    Check whether TE FP8 is usable by running a tiny Linear under an FP8 recipe.

    We deliberately use a small, "legal" shape for FP8 kernels:
      - batch = 16
      - last dim = 256 (multiple of 16)
    This should be safe across Hopper / Ada / Blackwell.
    """
    import torch

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, DelayedScaling
    except Exception as e:
        return "UNAVAILABLE", f"FP8 recipe import failed: {e}"

    # Construct a basic FP8 recipe (DelayedScaling with E4M3, as in TE docs)
    try:
        fp8_recipe = DelayedScaling(fp8_format=Format.E4M3)
    except Exception as e:
        return "UNAVAILABLE", f"Failed to create FP8 recipe: {e}"

    try:
        torch.cuda.empty_cache()

        in_features = 256
        out_features = 256

        lin = te.Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
        ).cuda()

        x = torch.randn(16, in_features, device="cuda", dtype=torch.bfloat16)

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y = lin(x)

        _ = y.detach()
    except Exception as e:
        msg = textwrap.shorten(str(e), width=200, placeholder="...")
        return "UNAVAILABLE", f"FP8 kernel path failed at runtime: {msg}"

    # If we get here without exceptions, FP8 via TE is fine for this setup.
    return "SUPPORTED", "FP8 recipe and kernel path ran successfully"


def check_nvfp4(device_name: str, major: int, minor: int, te_version: str):
    """
    NVFP4 check.

    We have three "interesting" situations:

      1) NVFP4 symbols not present            -> UNAVAILABLE
      2) NVFP4 present, but GPU is SM12x RTX  -> EXPERIMENTAL (we skip kernel test
                                                to avoid spamming PTX warnings)
      3) NVFP4 present on data-center Blackwell (SM100 family, etc.)
         -> we try a tiny kernel and call it SUPPORTED if it succeeds.

    The goal is to reflect reality *without* flooding stderr with messages like:
      "FP4 cvt PTX instructions are architecture-specific. Try recompiling with sm_XXXa..."
    on RTX Blackwell, where TE 2.8 isn't fully aligned yet.
    """
    import torch

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling, Format
    except Exception as e:
        return "UNAVAILABLE", f"Transformer Engine / NVFP4 import failed: {e}"

    # If NVFP4BlockScaling exists, NVFP4 is at least "known" to this TE build.
    try:
        nvfp4_recipe = NVFP4BlockScaling(fp4_format=Format.E2M1)
    except Exception as e:
        # Symbol exists but recipe construction fails -> treat as unavailable.
        return "UNAVAILABLE", f"Failed to create NVFP4 recipe: {e}"

    # Heuristic: SM12x "RTX ... Blackwell ..." is our current situation.
    # TE 2.8 exposes NVFP4 here, but kernels are not compiled for sm_120a/121a
    # in the 25.10 PyTorch container, causing a flood of PTX warnings.
    name_lower = device_name.lower()
    is_rtx_blackwell = (
        major >= 12
        and "rtx" in name_lower
        and "blackwell" in name_lower
    )

    if is_rtx_blackwell:
        # We deliberately *do not* run the kernel test here to avoid spam.
        # Mark as experimental / not fully supported until NVIDIA ships
        # a TE build compiled for sm_12xa NVFP4.
        return (
            "EXPERIMENTAL",
            "NVFP4 recipe present, but this GPU is RTX Blackwell (SM12x). "
            "Current NGC TE build is known to emit FP4 PTX warnings here; "
            "skipping NVFP4 kernel test and treating NVFP4 as experimental."
        )

    # For non-SM12x GPUs (e.g., data-center Blackwell or future architectures),
    # we try a tiny kernel and see if it works.
    try:
        torch.cuda.empty_cache()

        in_features = 256
        out_features = 256

        lin = te.Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
        ).cuda()

        # Choose shape compatible with FP8/NVFP4 constraints:
        # - product(dims[:-1]) divisible by 8 / 16
        # - last dim divisible by 16
        x = torch.randn(16, in_features, device="cuda", dtype=torch.bfloat16)

        with te.fp8_autocast(enabled=True, fp8_recipe=nvfp4_recipe):
            y = lin(x)

        _ = y.detach()
    except Exception as e:
        msg = textwrap.shorten(str(e), width=200, placeholder="...")
        return "UNAVAILABLE", f"NVFP4 kernel path failed at runtime: {msg}"

    return "SUPPORTED", "NVFP4 recipe and kernel path ran successfully"


def print_summary(
    te_version: str,
    fp8_status: str,
    fp8_reason: str,
    nvfp4_status: str,
    nvfp4_reason: str,
):
    print("=== Summary ===")
    print(f"Transformer Engine version: {te_version}")
    print()
    print(f"TE FP8 status:   {fp8_status}")
    print(f"  Details:       {fp8_reason}")
    print()
    print(f"NVFP4 status:    {nvfp4_status}")
    print(f"  Details:       {nvfp4_reason}")
    print()
    print("Notes:")
    print("  - This fork officially supports TE-FP8 inside the NGC PyTorch container")
    print("    on NVIDIA Blackwell-class GPUs (Hopper/Ada should also work).")
    print("  - NVFP4 is currently experimental on RTX Blackwell (SM12x) and may not")
    print("    be fully supported by the TE build shipped in the container.")
    print("  - For general users, BF16 + TE-FP8 is the recommended 'turbo' path;")
    print("    NVFP4 should be treated as a research-only option for now.")


if __name__ == "__main__":
    sys.exit(main())

