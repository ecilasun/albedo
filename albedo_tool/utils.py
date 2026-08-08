"""Image loading, saving, and batch processing utilities."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.bmp', '.jpg', '.jpeg', '.tiff', '.tif', '.webp'}


def load_image(path: str) -> Tuple[np.ndarray, Image.Image]:
    """Load an image from disk.

    Args:
        path: Path to the image file.

    Returns:
        Tuple of (numpy array [H, W, C] uint8, PIL Image).
    """
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    return arr, img


def save_image(
    arr: np.ndarray,
    path: str,
    output_format: Optional[str] = None,
    quality: int = 95,
) -> None:
    """Save a numpy array as an image.

    Args:
        arr: Image array, shape (H, W, C) or (H, W), dtype uint8 or float [0, 1].
        path: Output file path.
        output_format: Output format override (e.g. 'PNG', 'JPEG'). Auto-detected if None.
        quality: JPEG quality (1-100).
    """
    # Ensure uint8 [0, 255]
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (arr * 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Clamp values
    arr = np.clip(arr, 0, 255)

    output_format = {"JPG": "JPEG", "TIF": "TIFF"}.get(
        output_format,
        output_format,
    )
    img = Image.fromarray(arr)
    img.save(path, format=output_format, quality=quality, subsampling=0)


def get_image_paths(directory: str) -> List[str]:
    """Recursively find all image files in a directory.

    Args:
        directory: Path to the directory to scan.

    Returns:
        Sorted list of absolute paths to image files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(str(p) for p in dir_path.rglob(f'*{ext}'))
        images.extend(str(p) for p in dir_path.rglob(f'*{ext.upper()}'))

    return sorted(set(images))


def compute_resize_size(
    width: int,
    height: int,
    max_size: int,
) -> Tuple[int, int]:
    """Compute resized dimensions preserving aspect ratio.

    Args:
        width: Original width.
        height: Original height.
        max_size: Maximum dimension (width or height).

    Returns:
        Tuple (new_width, new_height).
    """
    if max(width, height) <= max_size:
        # Round to multiple of 32 for network compatibility
        new_w = max(32, (width // 32) * 32)
        new_h = max(32, (height // 32) * 32)
        return new_w, new_h

    ratio = max_size / max(width, height)
    new_w = max(32, int(width * ratio / 32) * 32)
    new_h = max(32, int(height * ratio / 32) * 32)
    return new_w, new_h


def image_to_tensor(
    arr: np.ndarray,
    resize: Optional[int] = None,
) -> np.ndarray:
    """Convert a PIL image array to a normalized tensor.

    Args:
        arr: Image array, shape (H, W, 3) or (H, W, C).
        resize: Maximum dimension for resizing. None for no resize.

    Returns:
        Tensor-like array, shape (3, H', W'), values in [0, 1].
    """
    # Convert to RGB if needed
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]  # Drop alpha
    elif arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)  # Grayscale to RGB

    # Resize if needed
    if resize is not None:
        h, w = arr.shape[:2]
        new_w, new_h = compute_resize_size(w, h, resize)
        pil_img = Image.fromarray(arr)
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32)

    # Normalize to [0, 1]
    arr = arr.astype(np.float32) / 255.0

    # Channel first: (H, W, 3) -> (3, H, W)
    arr = np.transpose(arr, (2, 0, 1))

    return arr


def tensor_to_image(tensor: np.ndarray) -> np.ndarray:
    """Convert a normalized tensor back to a uint8 image array.

    Args:
        tensor: Array, shape (3, H, W) or (H, W, 3), values in [0, 1].

    Returns:
        uint8 array, shape (H, W, 3).
    """
    # Channel last if needed
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))

    # Clamp and convert
    tensor = np.clip(tensor, 0.0, 1.0)
    return (tensor * 255.0).astype(np.uint8)


def build_output_path(
    input_path: str,
    output_dir: str,
    output_format: Optional[str] = None,
) -> str:
    """Build the output file path preserving directory structure.

    Args:
        input_path: Original input file path.
        output_dir: Base output directory.
        output_format: Output format extension override.

    Returns:
        Full output file path.
    """
    input_p = Path(input_path)
    suffix = f'.{output_format.lower()}' if output_format else input_p.suffix

    output_p = Path(output_dir) / (input_p.stem + suffix)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    return str(output_p)
