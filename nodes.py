from server import PromptServer

from .metrics import mae, mse, psnr, ssim


class ImagePSNR:
    """
    Peak Signal-to-Noise Ratio (dB) between two IMAGE batches, plus the
    underlying MSE it's computed from (a free byproduct, not a separate
    computation). Assumes the standard 0..1 IMAGE range (MAX=1) -
    deliberately not offered for SIGNED_IMAGE, since PSNR's MAX has to match
    the data's true peak-to-peak range, and getting that wrong silently
    shifts the result by a fixed amount (e.g. a -1..1 range needs MAX=2, not
    1).
    A batch is scored by computing PSNR per image pair and averaging the dB
    values - the same convention benchmark papers use over a whole test set
    - rather than exposing a batch_index to pick a single pair.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("psnr_db", "mse")
    FUNCTION = "execute"
    CATEGORY = "KULab/Metrics"
    OUTPUT_NODE = True  # runs even if outputs aren't wired to anything downstream

    def execute(self, image_a, image_b, unique_id=None):
        psnr_value = psnr(image_a, image_b)
        mse_value = mse(image_a, image_b)
        if unique_id is not None:
            PromptServer.instance.send_progress_text(f"PSNR: {psnr_value:.2f} dB\nMSE: {mse_value:.6f}", unique_id)
        return (psnr_value, mse_value)


class ImageMAE:
    """
    Mean Absolute Error between two IMAGE batches, averaged over the batch
    (per-image then averaged - equivalent to pooling every pixel together
    for this metric, since there's no log step to make the order matter).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("mae",)
    FUNCTION = "execute"
    CATEGORY = "KULab/Metrics"
    OUTPUT_NODE = True  # runs even if mae isn't wired to anything downstream

    def execute(self, image_a, image_b, unique_id=None):
        value = mae(image_a, image_b)
        if unique_id is not None:
            PromptServer.instance.send_progress_text(f"MAE: {value:.4f}", unique_id)
        return (value,)


class ImageSSIM:
    """
    Structural Similarity Index between two IMAGE batches, averaged over the
    batch (per-image then averaged, same convention as Image PSNR). Uses the
    standard Wang et al. 2004 defaults (11x11 Gaussian window, sigma=1.5,
    K1=0.01, K2=0.03) - no parameters are exposed, since these are what
    "SSIM" conventionally means in papers. Assumes the standard 0..1 IMAGE
    range, same reasoning as Image PSNR's MAX.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("ssim",)
    FUNCTION = "execute"
    CATEGORY = "KULab/Metrics"
    OUTPUT_NODE = True  # runs even if ssim isn't wired to anything downstream

    def execute(self, image_a, image_b, unique_id=None):
        value = ssim(image_a, image_b)
        if unique_id is not None:
            PromptServer.instance.send_progress_text(f"SSIM: {value:.4f}", unique_id)
        return (value,)


NODE_CLASS_MAPPINGS = {
    "KU_ImagePSNR": ImagePSNR,
    "KU_ImageMAE": ImageMAE,
    "KU_ImageSSIM": ImageSSIM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KU_ImagePSNR": "Image PSNR",
    "KU_ImageMAE": "Image MAE",
    "KU_ImageSSIM": "Image SSIM",
}