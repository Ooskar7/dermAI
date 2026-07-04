from __future__ import annotations

import sys

import numpy
import streamlit
import torch


def main() -> None:
    print(f"python: {sys.executable}")
    print(f"numpy: {numpy.__version__} ({numpy.__file__})")
    print(f"torch: {torch.__version__} ({torch.__file__})")
    print(f"streamlit: {streamlit.__version__} ({streamlit.__file__})")

    tensor = torch.from_numpy(numpy.zeros((1, 1), dtype=numpy.float32))
    print(f"torch/numpy interop: ok ({tuple(tensor.shape)})")

    major_version = int(numpy.__version__.split(".", maxsplit=1)[0])
    if major_version >= 2 and torch.__version__.startswith("2.2."):
        raise SystemExit(
            "This environment still has NumPy 2.x with Torch 2.2.x. "
            'Run: pip install --force-reinstall "numpy>=1.26,<2"'
        )


if __name__ == "__main__":
    main()
