#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data._utils.collate import default_collate


from dinos import DINOS

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="dataset", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--decoder_layers", default=[500, 500], type=int, nargs="+", help="Decoder layers.")
parser.add_argument("--encoder_layers", default=[500, 500], type=int, nargs="+", help="Encoder layers.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--generate_from", default=None, type=str, help="Generate images instead of training.")
parser.add_argument("--num_generate", default=1, type=int, help="How many images to generate.")
parser.add_argument("--save_to_dir", default=None, type=str, help="Path to save/load the model.")



def _collate_as_tuple(batch):
    collated = default_collate(batch)
    if isinstance(collated, torch.Tensor):
        return (collated,)
    return collated


# The VAE model
class VAE(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        #self.built = True
        self.learning_rate = 1e-3
        self._min_sd = 1e-6

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = torch.distributions.Normal(torch.zeros(args.z_dim), torch.ones(args.z_dim))

        # TODO: Define `self.encoder` as a `keras.Model`, which
        # - takes input images with shape `[MNIST.H, MNIST.W, MNIST.C]`
        # - flattens them
        # - applies `len(args.encoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.encoder_layers[i]` units
        # - generates two outputs `z_mean` and `z_sd`, each passing the result
        #   of the above bullet through its own dense layer of `args.z_dim` units,
        #   with `z_sd` using exponential function as activation to keep it positive.
        inputs = keras.layers.Input([DINOS.H, DINOS.W, DINOS.C])
        hidden = keras.layers.Flatten()(inputs)
        for l in args.encoder_layers:
            hidden = keras.layers.Dense(l, activation="relu")(hidden)

        z_mean = keras.layers.Dense(args.z_dim)(hidden)
        z_log_sd = keras.layers.Dense(args.z_dim)(hidden)
        
        self.encoder = keras.Model(inputs=inputs, outputs=(z_mean, z_log_sd))
         

        # TODO: Define `self.decoder` as a `keras.Model`, which
        # - takes vectors of `[args.z_dim]` shape on input
        # - applies `len(args.decoder_layers)` dense layers with ReLU activation,
        #   i-th layer with `args.decoder_layers[i]` units
        # - applies output dense layer with `MNIST.H * MNIST.W * MNIST.C` units
        #   and a suitable output activation
        # - reshapes the output (`keras.layers.Reshape`) to `[MNIST.H, MNIST.W, MNIST.C]`
        decoder_input = keras.layers.Input([args.z_dim])
        hidden2 = decoder_input
        for l in args.decoder_layers:
            hidden2 = keras.layers.Dense(l, activation="relu")(hidden2)
            
        hidden2 = keras.layers.Dense(DINOS.H * DINOS.W * DINOS.C, activation="sigmoid")(hidden2)
        hidden2 = keras.layers.Reshape([DINOS.H, DINOS.W, DINOS.C])(hidden2)

        self.decoder = keras.Model(inputs=decoder_input, outputs=hidden2)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.recon_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = keras.metrics.Mean(name="latent_loss")
        self._recon_loss = keras.losses.BinaryCrossentropy()

        
    def call(self, inputs, training=False):
        z_mean, z_log_sd = self.encoder(inputs, training=training)
        z_sd = F.softplus(z_log_sd) + self._min_sd
        q = torch.distributions.Normal(z_mean, z_sd)
        z = q.rsample() if training else z_mean
        return self.decoder(z, training=training)


    def train_step(self, data):
        # Keras may pass data as x or (x, y). Our loader returns (x, y).
        if isinstance(data, (tuple, list)):
            x = data[0]
            y = data[1] if len(data) > 1 else data[0]
        else:
            x = data
            y = data
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)

        self.zero_grad(set_to_none=True)

        z_mean, z_log_sd = self.encoder(x, training=True)
        z_sd = F.softplus(z_log_sd) + self._min_sd
        q = torch.distributions.Normal(z_mean, z_sd)
        z = q.rsample()
        x_pred = self.decoder(z, training=True)

        # Reconstruction loss
        recon = self._recon_loss(y, x_pred)

        # KL(q(z|x) || p(z))
        p = torch.distributions.Normal(torch.zeros_like(z_mean), torch.ones_like(z_sd))
        kl = torch.distributions.kl.kl_divergence(q, p).mean()

        loss = recon * (DINOS.H * DINOS.W * DINOS.C) + kl * self._z_dim

        loss.backward()

        trainable_weights = self.trainable_weights
        grads = [w.value.grad for w in trainable_weights]
        with torch.no_grad():
            self.optimizer.apply(grads, trainable_weights)

        # Update trackers (these are what Keras logs)
        self.loss_tracker.update_state(loss.detach())
        self.recon_tracker.update_state(recon.detach())
        self.kl_tracker.update_state(kl.detach())

        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.recon_tracker.result(),
            "latent_loss": self.kl_tracker.result(),
        }

    def generate(self, epoch: int, logs: dict[str, float]) -> None:
        GRID = 20

        # Generate GRIDxGRID images
        random_images = self.decoder(self._z_prior.sample([GRID * GRID]), training=False)

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = torch.stack([-2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
            ends = torch.stack([2 * torch.ones(GRID), torch.linspace(-2., 2., GRID)], -1)
        else:
            # Generate random Z
            starts = self._z_prior.sample([GRID])
            ends = self._z_prior.sample([GRID])
        interpolated_z = torch.cat(
            [starts[i] + (ends[i] - starts[i]) * torch.linspace(0., 1., GRID).unsqueeze(-1) for i in range(GRID)])
        interpolated_images = self.decoder(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated images
        image = torch.cat([
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(random_images, GRID)], axis=0),
            torch.zeros([DINOS.H * GRID, DINOS.W, DINOS.C]),
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(interpolated_images, GRID)], axis=0),
        ], axis=1)

def save_model(model: VAE, path: str):
    torch.save({
        "encoder": model.encoder.state_dict(),
        "decoder": model.decoder.state_dict(),
    }, path)

def load_model(model: VAE, path: str):
    checkpoint = torch.load(path, map_location="cpu")
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])

def main(args: argparse.Namespace) -> float:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create the network and train
    network = VAE(args)
    network.compile(optimizer=keras.optimizers.Adam())

    # If generation mode → load and generate
    if args.generate_from is not None:
        print(f"Loading model from {args.generate_from}")
        load_model(network, args.generate_from)

        with torch.no_grad():
            z = torch.randn(args.num_generate, args.z_dim)
            images = network.decoder(z, training=False)

        images = images.cpu().numpy()
        plt.imshow(images[0])
        plt.show()
        # np.save(args.output, images)
        # print(f"Generated {args.num_generate} images → saved to {args.output}")
        return 0.0

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    din = DINOS(args.dataset)
    train = torch.utils.data.DataLoader(
        din, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_as_tuple
    )

    logs = network.fit(train, epochs=args.epochs)

    if args.save_to_dir is not None:
        os.makedirs(args.save_to_dir, exist_ok=True)
        generator_path = os.path.join(args.save_to_dir, "vae_model.pt")
        print(f"Saving model to {generator_path}")
        save_model(network, generator_path)

    # Return loss for ReCodEx to validate
    return logs.history["loss"][-1]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
