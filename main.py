"""
Albedo Shadow Removal Tool

Usage:
    python main.py -i input.png -o output.png
    python main.py -i ./images/ -o ./outputs/ --resize 512 --verbose
    python main.py -i input.jpg -o output.jpg --model checkpoint.pth
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from albedo_tool.albedo_estimator import AlbedoEstimator
from albedo_tool.utils import load_image, save_image


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Albedo Shadow Removal Tool — Convert RGB images to shadow-free albedo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input image file or directory path.',
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file path or directory path.',
    )
    parser.add_argument(
        '-m', '--model',
        default=None,
           help='Path to a compatible local checkpoint. Defaults to the verified public '
               'Intrinsic_Decomposition checkpoint, downloaded once to the Hugging Face cache.',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on. Default: cpu.',
    )
    parser.add_argument(
        '--resize',
        type=int,
        default=None,
        help='Maximum dimension (width or height) for resizing before inference. '
             'Useful for speeding up processing on large images. Default: no resize.',
    )
    parser.add_argument(
        '--format',
        dest='output_format',
        default=None,
        choices=['png', 'jpg', 'bmp'],
        help='Output format. Defaults to the single output file extension or PNG for directories.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files.',
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print progress messages.',
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input path not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Check if input is a directory
    is_dir = os.path.isdir(args.input)

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available. Falling back to CPU.", file=sys.stderr)
        args.device = 'cpu'

    # Initialize estimator
    estimator = AlbedoEstimator(device=args.device)

    try:
        checkpoint_path = estimator.load_model(args.model)
        if args.verbose:
            print(f"Model loaded from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if is_dir:
            output_format = args.output_format or 'png'
            output_paths = estimator.estimate_albedo_batch(
                input_path=args.input,
                output_dir=args.output,
                resize=args.resize,
                output_format=output_format,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            if args.verbose:
                print(f"\nDone! {len(output_paths)} image(s) saved to: {args.output}")
        else:
            output_path = Path(args.output)
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing output: {output_path}")
                return

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_format = args.output_format or output_path.suffix.lstrip('.') or 'png'
            image, _ = load_image(args.input)
            albedo, _ = estimator.estimate_albedo(image, resize=args.resize)
            save_image(albedo, str(output_path), output_format=output_format.upper())
            if args.verbose:
                print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
