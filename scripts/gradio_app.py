#!/usr/bin/env python
"""Gradio UI launcher script - wrapper for discrete_diffusion.ui module."""

import sys
from pathlib import Path

# Add src to path for development (when not installed)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from discrete_diffusion.ui import main

if __name__ == "__main__":
    main()
