"""
Full-reference image quality metrics between two IMAGE tensors. Pure tensor
math, no ComfyUI node classes.
"""

import torch


def _per_image_mse(image_a: torch.Tensor, image_b: torch.Tensor) -> torch.Tensor:
    # image_a/image_b: [B,H,W,C]. Returns [B]: mean squared error per batch element.
    diff = (image_a - image_b).float()
    return diff.pow(2).flatten(1).mean(dim=1)


def _check_same_shape(image_a: torch.Tensor, image_b: torch.Tensor):
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"image_a and image_b must have the same shape, got "
            f"{tuple(image_a.shape)} and {tuple(image_b.shape)}"
        )


def psnr(image_a: torch.Tensor, image_b: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio, in dB, averaged over the batch.
    Computed per batch element first, then averaged - the standard
    benchmark-paper convention (average the dB values across a set of
    images) rather than pooling every pixel from every image into one
    giant MSE, which would give a different number due to the log.
    A tiny epsilon keeps identical images (MSE=0) from producing +inf.
    Assumes image_a/image_b are already in the same range as max_val
    (default 1.0, matching ComfyUI's IMAGE convention).
    """
    _check_same_shape(image_a, image_b)

    mse = _per_image_mse(image_a, image_b).clamp(min=1e-10)
    per_image_db = 10.0 * torch.log10(max_val ** 2 / mse)
    return per_image_db.mean().item()


def mae(image_a: torch.Tensor, image_b: torch.Tensor) -> float:
    """
    Mean Absolute Error, averaged over the batch. Unlike psnr(), there's no
    log step here, so per-image-then-average and pooling-all-pixels give the
    identical result - this still computes it per-image for symmetry.
    """
    _check_same_shape(image_a, image_b)

    diff = (image_a - image_b).float().abs()
    per_image = diff.flatten(1).mean(dim=1)
    return per_image.mean().item()


def mse(image_a: torch.Tensor, image_b: torch.Tensor) -> float:
    """
    Mean Squared Error, averaged over the batch (per-image then averaged;
    no log step, so this is the same either way). This is the same
    per-image quantity psnr() computes internally before taking the log -
    exposed separately so Image PSNR can report it too without duplicating
    psnr()'s formula.
    """
    _check_same_shape(image_a, image_b)
    return _per_image_mse(image_a, image_b).mean().item()


# Standard SSIM parameters from Wang et al. 2004 - not exposed as node
# parameters, since these defaults are what "SSIM" conventionally means in
# the literature.
_SSIM_WINDOW_SIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_K1 = 0.01
_SSIM_K2 = 0.03


def _gaussian_window(window_size: int, sigma: float, channels: int, dtype, device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d /= g_1d.sum()
    g_2d = torch.outer(g_1d, g_1d)  # [window_size, window_size], sums to 1
    return g_2d.expand(channels, 1, window_size, window_size).to(dtype=dtype, device=device)


def ssim(image_a: torch.Tensor, image_b: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Structural Similarity Index (Wang et al. 2004), averaged over the batch.
    Uses a fixed 11x11 Gaussian window (sigma=1.5) and the standard
    stabilization constants K1=0.01, K2=0.03 - no parameters are exposed.
    Each channel is filtered independently (depthwise convolution), then the
    local SSIM map is averaged over H, W, and C together (same
    channel-combining convention as psnr()/mae()); batch elements are scored
    independently and averaged, same convention as psnr().
    """
    _check_same_shape(image_a, image_b)

    _, _, _, c = image_a.shape
    x = image_a.float().permute(0, 3, 1, 2)  # [B,C,H,W]
    y = image_b.float().permute(0, 3, 1, 2)

    window = _gaussian_window(_SSIM_WINDOW_SIZE, _SSIM_SIGMA, c, x.dtype, x.device)
    pad = _SSIM_WINDOW_SIZE // 2

    def blur(t):
        return torch.nn.functional.conv2d(t, window, padding=pad, groups=c)

    mu_x, mu_y = blur(x), blur(y)
    mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x_sq = blur(x * x) - mu_x_sq
    sigma_y_sq = blur(y * y) - mu_y_sq
    sigma_xy = blur(x * y) - mu_xy

    c1 = (_SSIM_K1 * max_val) ** 2
    c2 = (_SSIM_K2 * max_val) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))

    per_image = ssim_map.flatten(1).mean(dim=1)
    return per_image.mean().item()