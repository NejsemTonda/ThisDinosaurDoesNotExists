#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import pickle
import matplotlib.pyplot as plt
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np
import torch
torch.set_default_dtype(torch.float16)

from dinos import DINOS

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--dataset", default="dataset", type=str, help="MNIST-like dataset to use.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--train_size", default=None, type=int, help="Limit on the train set size.")
parser.add_argument("--z_dim", default=100, type=int, help="Dimension of Z.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--generate_images", default=False, type=bool, help="Path to saved generator model.")
parser.add_argument("--num_generate", default=1, type=int, help="Number of images to generate.")
parser.add_argument("--save_model", default=True, type=bool, help="Indicator, whether the model should be saved.")
parser.add_argument("--resume_training", default=False, type=bool, help="Path to checkpoint to continue training from.")



# The GAN model
class GAN(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self._seed = args.seed
        self._z_dim = args.z_dim
        self._z_prior = torch.distributions.Normal(torch.zeros(args.z_dim), torch.ones(args.z_dim))

        # TODO: Define `self.generator` as a `keras.Model`, which
        # - takes vectors of shape `[args.z_dim]` on input
        # - applies batch normalized dense layer with 1_024 units and ReLU
        # - applies batch normalized dense layer with `MNIST.H // 4 * MNIST.W // 4 * 64` units and ReLU
        # - reshapes the current hidden output to `[MNIST.H // 4, MNIST.W // 4, 64]`
        # - applies batch normalized transposed convolution with 32 filters, kernel size 4,
        #   stride 2, same padding, and ReLU activation
        # - applies transposed convolution with `MNIST.C` filters, kernel size 4,
        #   stride 2, same padding, and a suitable output activation
        # Note that on the lecture, we discussed that layers before batch normalization should
        # not use bias -- but for simplicity, do not do it here (so do not set `use_bias=False`).
        inputs_generator = keras.layers.Input([args.z_dim])

        hidden_generator = keras.layers.Dense(1024)(inputs_generator)
        hidden_generator = keras.layers.BatchNormalization()(hidden_generator)
        hidden_generator = keras.layers.ReLU()(hidden_generator)

        hidden_generator = keras.layers.Dense(DINOS.H // 4 * DINOS.W // 4 * 64)(hidden_generator)
        hidden_generator = keras.layers.BatchNormalization()(hidden_generator)
        hidden_generator = keras.layers.ReLU()(hidden_generator)

        hidden_generator = keras.layers.Reshape([DINOS.H // 4, DINOS.W // 4, 64])(hidden_generator)

        hidden_generator = keras.layers.Conv2DTranspose(32, 4, 2, padding='same')(hidden_generator)
        hidden_generator = keras.layers.BatchNormalization()(hidden_generator)
        hidden_generator = keras.layers.ReLU()(hidden_generator)

        outputs_generator = keras.layers.Conv2DTranspose(DINOS.C, 4, 2, padding='same', activation='sigmoid')(hidden_generator)

        self.generator = keras.Model(inputs=inputs_generator, outputs=outputs_generator)

        # TODO: Define `self.discriminator` as a `keras.Model`, which
        # - takes input images with shape `[MNIST.H, MNIST.W, MNIST.C]`
        # - computes batch normalized convolution with 32 filters, kernel size 5,
        #   same padding, and ReLU activation.
        # - max-pools with pool size 2 and stride 2
        # - computes batch normalized convolution with 64 filters, kernel size 5,
        #   same padding, and ReLU activation
        # - max-pools with pool size 2 and stride 2
        # - flattens the current representation
        # - applies batch normalized dense layer with 1_024 units and ReLU activation
        # - applies output dense layer with one output and a suitable activation function
        inputs_discriminator = keras.layers.Input([DINOS.H, DINOS.W, DINOS.C])

        hidden_discriminator = keras.layers.Conv2D(32, 5, padding='same')(inputs_discriminator)
        hidden_discriminator = keras.layers.BatchNormalization()(hidden_discriminator)
        hidden_discriminator = keras.layers.ReLU()(hidden_discriminator)

        hidden_discriminator = keras.layers.MaxPooling2D(2, 2)(hidden_discriminator)

        hidden_discriminator = keras.layers.Conv2D(64, 5, padding='same')(hidden_discriminator)
        hidden_discriminator = keras.layers.BatchNormalization()(hidden_discriminator)
        hidden_discriminator = keras.layers.ReLU()(hidden_discriminator)

        hidden_discriminator = keras.layers.MaxPooling2D(2, 2)(hidden_discriminator)

        hidden_discriminator = keras.layers.Flatten()(hidden_discriminator)

        hidden_discriminator = keras.layers.Dense(1024)(hidden_discriminator)
        hidden_discriminator = keras.layers.BatchNormalization()(hidden_discriminator)
        hidden_discriminator = keras.layers.ReLU()(hidden_discriminator)

        outputs_discriminator = keras.layers.Dense(1, activation='sigmoid')(hidden_discriminator)

        self.discriminator = keras.Model(inputs=inputs_discriminator, outputs=outputs_discriminator)


    # We override `compile`, because we want to use two optimizers.
    def compile(
        self, discriminator_optimizer: keras.optimizers.Optimizer, generator_optimizer: keras.optimizers.Optimizer,
        loss: keras.losses.Loss, metric: keras.metrics.Metric,
    ) -> None:
        super().compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss = loss
        self.metric = metric
        self.built = True

    def train_step(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # TODO(gan): Generator training.
        # - generate as many random latent samples as there are `images`, by a single call
        #   to `self._z_prior.sample`;
        # - pass the samples through a generator; do not forget about `training=True`
        # - run discriminator on the generated images, also using `training=True` (even if
        #   not updating discriminator parameters, we want to perform possible BatchNormalization in it)
        # - compute `generator_loss` using `self.loss`, with ones as target labels
        #   (`torch.ones_like` might come handy).
        # Then, run an optimizer step with respect to generator trainable variables.
        # Do not forget that we created `generator_optimizer` in the `compile` override.
        self.generator.zero_grad()
        z = self._z_prior.sample([images.shape[0]])
        pred_imgs = self.generator(z, training=True)

        discr_pred = self.discriminator(pred_imgs, training=True)

        generator_loss = self.loss(torch.ones_like(discr_pred), discr_pred)
        generator_loss.backward()
        generator_grads = [var.value.grad for var in self.generator.trainable_variables]
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_variables))



        # TODO(gan): Discriminator training. Using a Gradient tape:
        # - discriminate `images` with `training=True`, storing
        #   results in `discriminated_real`
        # - discriminate images generated in generator training with `training=True`,
        #   storing results in `discriminated_fake`
        # - compute `discriminator_loss` by summing
        #   - `self.loss` on `discriminated_real` with suitable targets,
        #   - `self.loss` on `discriminated_fake` with suitable targets.
        # Then, run an optimizer step with respect to discriminator trainable variables.
        # Do not forget that we created `discriminator_optimizer` in the `compile` override.
        self.discriminator.zero_grad()
        discriminated_real = self.discriminator(images, training=True)
        discriminated_fake = self.discriminator(pred_imgs.detach(), training=True)
        
        real_loss = self.loss(torch.ones_like(discriminated_real), discriminated_real)
        fake_loss = self.loss(torch.zeros_like(discriminated_fake), discriminated_fake)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()

        discriminator_grads = [var.value.grad for var in self.discriminator.trainable_variables]
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))
        

        # TODO(gan): Update the discriminator accuracy metric -- call the
        # `self.metric` twice, with the same arguments the `self.loss`
        # was called during discriminator loss computation.
        self.metric(torch.ones_like(discriminated_real), discriminated_real)
        self.metric(torch.zeros_like(discriminated_fake), discriminated_fake)



        self._loss_tracker.update_state(discriminator_loss + generator_loss)
        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
            **self.get_metrics_result(),
        }

    def generate(self, epoch: int, logs: dict[str, torch.Tensor]) -> None:
        GRID = 20

        device = next(self.generator.parameters()).device

        # Generate GRIDxGRID images
        random_images = self.generator(self._z_prior.sample([GRID * GRID]).to(device), training=False)

        # Generate GRIDxGRID interpolated images
        if self._z_dim == 2:
            # Use 2D grid for sampled Z
            starts = torch.stack([-2 * torch.ones(GRID, device=device), torch.linspace(-2., 2., GRID, device=device)], -1)
            ends = torch.stack([2 * torch.ones(GRID, device=device), torch.linspace(-2., 2., GRID, device=device)], -1)
        else:
            # Generate random Z
            starts = self._z_prior.sample([GRID]).to(device)
            ends = self._z_prior.sample([GRID]).to(device)
        interpolated_z = torch.cat(
            [starts[i] + (ends[i] - starts[i]) * torch.linspace(0., 1., GRID, device=device).unsqueeze(-1) for i in range(GRID)])
        interpolated_images = self.generator(interpolated_z, training=False)

        # Stack the random images, then an empty row, and finally interpolated images
        image = torch.cat([
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(random_images, GRID)], axis=0),
            torch.zeros([DINOS.H * GRID, DINOS.W, DINOS.C], device=device),
            torch.cat([torch.cat(list(images), axis=1) for images in torch.chunk(interpolated_images, GRID)], axis=0),
        ], axis=1)

def load_generator_and_generate(path: str, num_images: int, z_dim: int):
    generator = keras.models.load_model(path)

    z = torch.randn([num_images, z_dim])
    images = generator(z, training=False)

    return images

def save_checkpoint(network: GAN, path: str):
    torch.save({
        "generator": network.generator.state_dict(),
        "discriminator": network.discriminator.state_dict(),
    }, path)


def load_checkpoint(network: GAN, path: str):
    checkpoint = torch.load(path, map_location="cpu")
    network.generator.load_state_dict(checkpoint["generator"])
    network.discriminator.load_state_dict(checkpoint["discriminator"])


def main(args: argparse.Namespace) -> dict[str, float]:
    model_dir = os.path.join("models", "dcgan", "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    if args.generate_images:
        generator_path = os.path.join(model_dir, "generator.keras")
        imgs = load_generator_and_generate(generator_path, args.num_generate, args.z_dim)
        imgs = imgs.detach().numpy()
        for img in imgs:
            plt.imshow(img)
            plt.show()
        print(f"Generated images shape: {imgs.shape}")
        return {}
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    dinos = DINOS(args.dataset)
    train = torch.utils.data.DataLoader(dinos, batch_size=args.batch_size, shuffle=True)


    # Create the network and train
    network = GAN(args)
    network.compile(
        discriminator_optimizer=keras.optimizers.Adam(),
        generator_optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metric=keras.metrics.BinaryAccuracy("discriminator_accuracy"),
    )

    if args.resume_training:
        resume_from = os.path.join(model_dir, "dcgan_model.pt")
        if os.path.exists(resume_from):
            print(f"Resuming training from {resume_from}")
            load_checkpoint(network, resume_from)
        else:
            print(f"No file to resume learning from.")

    logs = network.fit(train, epochs=args.epochs)

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    with open(f"{args.logdir}.pkl", "wb") as log_file:
        pickle.dump(logs.history, log_file)
    

    # Save trained generator model
    if args.save_model:
        os.makedirs(model_dir, exist_ok=True)

        generator_path = os.path.join(model_dir, "generator.keras")
        checkpoint_path = os.path.join(model_dir, "dcgan_model.pt")

        network.generator.save(generator_path)
        save_checkpoint(network, checkpoint_path)

        print(f"Generator saved to {generator_path}")
        print(f"Checkpoint saved to {checkpoint_path}")

    # Return the loss and the discriminator accuracy for ReCodEx to validate.
    return {metric: logs.history[metric][-1] for metric in ["loss", "discriminator_accuracy"]}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
