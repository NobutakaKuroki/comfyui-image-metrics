"""
@author: Dr. Nobutaka Kuroki (Kobe University)
@title: ComfyUI Image Metrics
@description: Full-reference image quality metrics (PSNR, MAE, ...) between two IMAGE batches, shown directly on the node and output as FLOAT for further use.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]