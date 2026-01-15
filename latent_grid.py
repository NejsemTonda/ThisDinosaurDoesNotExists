#!/usr/bin/env python3
"""
Generate an N x N grid of images by sampling the VAE latent space.

For z_dim=2, creates a uniform 2D grid from -range to +range.
For z_dim>2, interpolates along the first two latent dimensions while keeping others at 0.
"""
import argparse
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np
from PIL import Image
import re

from dinos import DINOS

parser = argparse.ArgumentParser()
parser.add_argument("--grid_size", "-n", default=9, type=int, help="Grid size N (creates NxN images)")
parser.add_argument("--z_dim", default=2, type=int, help="Latent space dimension (must match model)")
parser.add_argument("--range", default=10.0, type=float, help="Range for latent space sampling [-range, range]")
parser.add_argument("--model_path", default="models\\vae\\vae.py-bs=100,d=dataset,dl=[500, 500],el=[500, 500],zd=2\\vae_model_3.pt", type=str, help="Path to VAE model checkpoint")
parser.add_argument("--output", "-o", default="latent_grid.png", type=str, help="Output PNG filename")
parser.add_argument("--encoder_layers", default=[500, 500], type=int, nargs="+", help="Encoder layers (must match model)")
parser.add_argument("--decoder_layers", default=[500, 500], type=int, nargs="+", help="Decoder layers (must match model)")
parser.add_argument("--seed", default=122, type=int, help="Random seed")


def main(args):
    import keras
    keras.utils.set_random_seed(args.seed)

    # Import VAE class
    from vae import VAE, load_model

    # Create model with matching architecture
    network = VAE(args)

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Use default path based on parameters
        import re
        args_to_be_mentioned = ["batch_size", "dataset", "decoder_layers", "encoder_layers", "z_dim"]
        # Set defaults for args that might not be provided
        if not hasattr(args, 'batch_size'):
            args.batch_size = 50
        if not hasattr(args, 'dataset'):
            args.dataset = "dataset"

        model_dir = os.path.join("models", "vae", "{}-{}".format(
            "vae.py",
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                      for k, v in sorted(filter(lambda kv: kv[0] in args_to_be_mentioned, vars(args).items()))))
        ))
        model_path = os.path.join(model_dir, "vae_model_4.pt")

    print(f"Loading model from {model_path}")
    load_model(network, model_path)

    N = args.grid_size
    r = args.range

    # Create grid of latent vectors
    # For z_dim >= 2, vary first two dimensions in a grid, rest are zeros
    z_grid = torch.zeros(N * N, args.z_dim)

    # Create linear spaces for the two dimensions
    lin = torch.linspace(-r, r, N)

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            z_grid[idx, 0] = lin[j]  # x-axis: first latent dim
            z_grid[idx, 1] = lin[N - 1 - i]  # y-axis: second latent dim (inverted for image coords)

    print(f"Generating {N}x{N} = {N*N} images...")

    # Generate images
    with torch.no_grad():
        images = network.decoder(z_grid, training=False)

    images = images.cpu().numpy()

    # Assemble grid image
    H, W, C = DINOS.H, DINOS.W, DINOS.C
    grid_image = np.zeros((N * H, N * W, C), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            idx = i * N + j
            grid_image[i * H:(i + 1) * H, j * W:(j + 1) * W, :] = images[idx]

    # Convert to uint8 and save
    grid_image = (grid_image * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(grid_image)
    img.save(args.output)

    print(f"Saved grid image to {args.output}")


if __name__ == "__main__":
    args = parser.parse_args()
    # Set defaults needed by VAE constructor
    args.batch_size = getattr(args, 'batch_size', 50)
    args.dataset = getattr(args, 'dataset', 'dataset')
    main(args)
