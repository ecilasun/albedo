# Albedo Shadow Removal Tool

A Python tool that converts RGB images (PNG, BMP, JPG, etc.) to **albedo** (shadow-free reflectance) images using a pretrained intrinsic-decomposition model.

## Features

- **Pretrained intrinsic decomposition** with automatic, pinned checkpoint download
- **Batch processing** — process entire directories of images
- **Multiple image formats** — PNG, BMP, JPG, JPEG, TIFF, WebP
- **CPU-friendly** — runs on CPU without GPU
- **Simple CLI** — easy to use from command line or scripts

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

Create virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `torch` — PyTorch for model inference
- `torchvision` — ResNet-18 encoder used by the checkpoint
- `Pillow` — Image loading/saving
- `numpy` — Array operations
- `tqdm` — Progress bars
- `huggingface_hub` — verified checkpoint download and caching

## Usage

### Single Image

```bash
python main.py -i input.png -o output.png
```

### Directory (Batch Processing)

```bash
python main.py -i ./images/ -o ./outputs/
```

### With Options

```bash
# Resize large images for faster processing
python main.py -i input.jpg -o output.jpg --resize 512

# Specify output format
python main.py -i input.png -o output.jpg --format jpg

# Overwrite existing outputs
python main.py -i ./images/ -o ./outputs/ --overwrite

# Verbose output
python main.py -i input.png -o output.png -v
```

### Model Selection

```bash
# Downloads the default checkpoint on first use, then runs from the local cache.
python main.py -i input.png -o output.png

# Use a local copy of that compatible checkpoint instead.
python main.py -i input.png -o output.png -m path/to/model_final.pth
```

## Command-Line Arguments

| Argument | Description | Required |
| -------- | ----------- | -------- |
| `-i, --input` | Input image file or directory path | Yes |
| `-o, --output` | Output file or directory path | Yes |
| `-m, --model` | Path to a compatible local checkpoint; otherwise downloads the default | No |
| `--device` | Device: `cpu` or `cuda` (default: cpu) | No |
| `--resize` | Max dimension for resizing before inference | No |
| `--format` | Output format: `png`, `jpg`, `bmp`; defaults to output extension for one file or PNG for a directory | No |
| `--overwrite` | Overwrite existing output files | No |
| `-v, --verbose` | Print progress messages | No |

## How It Works

The tool uses a ResNet-18 encoder with separate decoder heads to decompose an input image into two components:

1. **Albedo** — The true surface color (reflectance), free from shadows and lighting variations
2. **Illumination** — The lighting/shadow component

Based on the **Retinex theory**: `Image = Albedo × Illumination`

Large images are processed in overlapping 512 px tiles, which keeps inference comfortably within a 16 GB GPU. CPU inference is also supported, though slower.

## Project Structure

```text
albedo/
├── albedo_tool/
│   ├── __init__.py              # Package init
│   ├── albedo_estimator.py      # Core inference engine
│   └── utils.py                 # Image I/O utilities
├── models/
│   └── intrinsic_decomposition.py # Checkpoint-compatible model architecture
├── main.py                      # CLI entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Pretrained Weights

The default is [`ssy1245/Intrinsic_Decomposition`](https://huggingface.co/ssy1245/Intrinsic_Decomposition), revision `da2b229a626c617795cc25c34bdc5a8ac3813cb9`, file `full_v4/model_final.pth` (58.6 MB, MIT license). It is downloaded once by `huggingface_hub` into the standard local Hugging Face cache and is loaded with `strict=True`; unrelated `.pth` files are rejected rather than partially loaded.

## Performance Tips

- **Resize large images**: Use `--resize 512` to downscale before inference for faster processing
- **PNG output**: Use PNG for lossless output; JPG for smaller file sizes
- **Batch processing**: The tool skips already-processed files; use `--overwrite` to reprocess

## Limitations

- **CPU inference**: Supported but slow for large images
- **Domain fit**: The checkpoint was trained for intrinsic decomposition; it is not a guaranteed material scanner for arbitrary rendered or synthetic textures
- **Complex shadows**: Very complex or soft shadows may not be fully removed
- **Output resolution**: `--resize` changes the output resolution as well as the inference resolution

## License

This project is provided as-is for educational and research purposes.

## References

- [Intrinsic_Decomposition model and checkpoint](https://huggingface.co/ssy1245/Intrinsic_Decomposition)
- [Retinex Theory](https://en.wikipedia.org/wiki/Retinex)
