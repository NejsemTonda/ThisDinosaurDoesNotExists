---
title: This Dinosaur Does Not Exist
subtitle: "Year Project"
author: "Matej Kukurugya & Václav Krňák"
date: \today
documentclass: beamer
theme: Madrid
colortheme: beaver
---

[//]: <> (This is a comment and is not visible in the presentation)
[//]: <> (make cmd: pandoc presentaion.md -t beamer -o presentaion.pdf)


# What We Wanted to Achieve

## Inspiration

https://thispersondoesnotexist.com/

![thisperson](src/thisperson.jpg)

# What Was Needed

::: incremental

* Create a model that would generate images of dinosaurs
    * "State-of-the-art"
    * Explore and compare different models
* Use of the school's HPC cluster for training on graphics cards
* Creation of a web application

:::

# Data Collection

To create training data, we used https://www.kaggle.com/. Specifically:

* https://www.kaggle.com/datasets/larserikrisholm/dinosaur-image-dataset-15-species – 2448 images - 510MB
* https://www.kaggle.com/datasets/cmglonly/simple-dinosurus-dataset - 200 images - 58MB
* https://www.kaggle.com/datasets/antaresl/jurassic-park-dinosaurs-dataset - 4364 images - 4GB 
* https://www.kaggle.com/datasets/caokhoihuynh/jurassic-world-dinosaur - 5835 images - 1.6GB

# Pre-processing

* Images are rescaled to 256x256 px
* Given the size of the dataset (12848 images total), we decided not to include any augmentation techniques

# Model Selection

## Stable Diffusion

* Iterative noise removal
* Too computationally demanding for our use

## DC-GAN

* The model has two parts: Generator and discriminator
* Too difficult a task, the discriminator always achieves 100% accuracy

## VAE

* Encoder -> Latent space -> Decoder
* Computationally undemanding

# Training

![z_dim](src/zdim_loss.svg){ width=11cm } 

# Results 

- 48 epochs

| 150 | 50 | 10 | 5 | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_0_0.png) | ![](./src/gen_images/vae/vae_50_0_13.png) | ![](./src/gen_images/vae/vae_10_0_1.png) | ![](./src/gen_images/vae/vae_5_0_43.png) | ![](./src/gen_images/vae/vae_2_0_6.png) |
| ![](./src/gen_images/vae/vae_150_0_7.png) | ![](./src/gen_images/vae/vae_50_0_44.png) | ![](./src/gen_images/vae/vae_10_0_45.png) | ![](./src/gen_images/vae/vae_5_0_55.png) | ![](./src/gen_images/vae/vae_2_0_20.png) |
| ![](./src/gen_images/vae/vae_150_0_30.png) | ![](./src/gen_images/vae/vae_50_0_36.png) | ![](./src/gen_images/vae/vae_10_0_58.png) | ![](./src/gen_images/vae/vae_5_0_69.png) | ![](./src/gen_images/vae/vae_2_0_27.png) |

# Results 

- 96 epochs

| 150 | 50 | 10 | 5 | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_1_0.png) | ![](./src/gen_images/vae/vae_50_1_13.png) | ![](./src/gen_images/vae/vae_10_1_1.png) | ![](./src/gen_images/vae/vae_5_1_43.png) | ![](./src/gen_images/vae/vae_2_1_6.png) |
| ![](./src/gen_images/vae/vae_150_1_7.png) | ![](./src/gen_images/vae/vae_50_1_44.png) | ![](./src/gen_images/vae/vae_10_1_45.png) | ![](./src/gen_images/vae/vae_5_1_55.png) | ![](./src/gen_images/vae/vae_2_1_20.png) |
| ![](./src/gen_images/vae/vae_150_1_30.png) | ![](./src/gen_images/vae/vae_50_1_36.png) | ![](./src/gen_images/vae/vae_10_1_58.png) | ![](./src/gen_images/vae/vae_5_1_69.png) | ![](./src/gen_images/vae/vae_2_1_27.png) |

# Results 

- 144 epochs

| 150 | 50 | 10 | 5 | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_2_0.png) | ![](./src/gen_images/vae/vae_50_2_13.png) | ![](./src/gen_images/vae/vae_10_2_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_2_6.png) |
| ![](./src/gen_images/vae/vae_150_2_7.png) | ![](./src/gen_images/vae/vae_50_2_44.png) | ![](./src/gen_images/vae/vae_10_2_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_2_20.png) |
| ![](./src/gen_images/vae/vae_150_2_30.png) | ![](./src/gen_images/vae/vae_50_2_36.png) | ![](./src/gen_images/vae/vae_10_2_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_2_27.png) |

# Results 

- 192 epochs

| 150 | 50 | 10 | 5 (144 ep.) | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_3_0.png) | ![](./src/gen_images/vae/vae_50_3_13.png) | ![](./src/gen_images/vae/vae_10_3_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_3_6.png) |
| ![](./src/gen_images/vae/vae_150_3_7.png) | ![](./src/gen_images/vae/vae_50_3_44.png) | ![](./src/gen_images/vae/vae_10_3_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_3_20.png) |
| ![](./src/gen_images/vae/vae_150_3_30.png) | ![](./src/gen_images/vae/vae_50_3_36.png) | ![](./src/gen_images/vae/vae_10_3_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_3_27.png) |

# Results 

- 240 epochs

| 150 | 50 | 10 | 5 (144 ep.) | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_4_0.png) | ![](./src/gen_images/vae/vae_50_4_13.png) | ![](./src/gen_images/vae/vae_10_4_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_4_6.png) |
| ![](./src/gen_images/vae/vae_150_4_7.png) | ![](./src/gen_images/vae/vae_50_4_44.png) | ![](./src/gen_images/vae/vae_10_4_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_4_20.png) |
| ![](./src/gen_images/vae/vae_150_4_30.png) | ![](./src/gen_images/vae/vae_50_4_36.png) | ![](./src/gen_images/vae/vae_10_4_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_4_27.png) |

# Results 

- 288 epochs

| 150 | 50 | 10 | 5 (144 ep.) | 2 (240 ep.) |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_5_0.png) | ![](./src/gen_images/vae/vae_50_5_13.png) | ![](./src/gen_images/vae/vae_10_5_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_4_6.png) |
| ![](./src/gen_images/vae/vae_150_5_7.png) | ![](./src/gen_images/vae/vae_50_5_44.png) | ![](./src/gen_images/vae/vae_10_5_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_4_20.png) |
| ![](./src/gen_images/vae/vae_150_5_30.png) | ![](./src/gen_images/vae/vae_50_5_36.png) | ![](./src/gen_images/vae/vae_10_5_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_4_27.png) |

# Results

![](./latent_grid.png){ height=8cm }

# Discussion

::: incremental

* The models trained properly, but did not have sufficient capacity
* Models with large latent space were a dead end


:::
