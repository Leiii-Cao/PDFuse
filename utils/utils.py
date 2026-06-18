import torch
import torch.nn.functional as F


def rgb01_to_ycbcr01(rgb):
    """Convert an RGB image tensor from [0, 1] range to YCbCr in [0, 1]."""
    red = rgb[:, 0, :, :]
    green = rgb[:, 1, :, :]
    blue = rgb[:, 2, :, :]

    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 0.5
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 0.5

    return torch.stack([y, cb, cr], dim=1)


def ycbcr01_to_rgb01(ycbcr):
    """Convert a YCbCr image tensor from [0, 1] range to RGB in [0, 1]."""
    y = ycbcr[:, 0, :, :]
    cb = ycbcr[:, 1, :, :]
    cr = ycbcr[:, 2, :, :]

    red = y + 1.402 * (cr - 0.5)
    green = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    blue = y + 1.772 * (cb - 0.5)

    return torch.stack([red, green, blue], dim=1).clamp(0, 1)


def rgb_to_ycbcr_tensor(rgb):
    """Convert an RGB image tensor from [-1, 1] range to YCbCr in [-1, 1]."""
    return rgb01_to_ycbcr01((rgb + 1) / 2.0) * 2 - 1


def ycbcr_to_rgb_tensor(ycbcr):
    """Convert a YCbCr image tensor from [-1, 1] range to RGB in [-1, 1]."""
    return ycbcr01_to_rgb01((ycbcr + 1) / 2.0) * 2 - 1


def make_sobel_kernels(device, channels):
    """Create depthwise Sobel kernels for horizontal and vertical gradients."""
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=device,
    )
    kernel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=device,
    )
    kernel_x = kernel_x.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel_x, kernel_y


def make_laplacian_kernel(device, channels):
    """Create a depthwise 3x3 Laplacian kernel."""
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=torch.float32,
        device=device,
    )
    return kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def depthwise_conv2d(image, kernel, padding=1):
    """Apply the same spatial convolution independently to each channel."""
    return F.conv2d(image, kernel, padding=padding, groups=image.shape[1])


def image_gradients(image, kernel_x, kernel_y):
    """Compute x and y image gradients with the provided depthwise kernels."""
    grad_x = depthwise_conv2d(image, kernel_x, padding=1)
    grad_y = depthwise_conv2d(image, kernel_y, padding=1)
    return grad_x, grad_y


def make_gaussian_kernel(kernel_size=11, sigma=5, channels=3):
    """Create a depthwise 2D Gaussian blur kernel."""
    coord = torch.arange(kernel_size) - kernel_size // 2
    gaussian_1d = torch.exp(-(coord**2) / (2 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
    return gaussian_2d.repeat(channels, 1, 1, 1)


def gaussian_blur(image, kernel_size=11, sigma=5):
    """Apply depthwise Gaussian blur to a BCHW image tensor."""
    channels = image.shape[1]
    kernel = make_gaussian_kernel(kernel_size, sigma, channels).to(image.device, image.dtype)
    return F.conv2d(image, kernel, groups=channels, padding=kernel_size // 2)
