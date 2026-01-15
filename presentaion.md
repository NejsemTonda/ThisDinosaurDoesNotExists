---
title: This Dinosaur Does Not Exists
subtitle: "Ročníkový projekt"
author: "Matej Kukurugya & Václav Krňák"
date: \today
documentclass: beamer
theme: Madrid
colortheme: beaver
---

[//]: <> (Tohle je komentář a není vidět v prezentaci)
[//]: <> (make cmd: pandoc presentaion.md -t beamer -o presentaion.pdf)


# Čeho jsme chtěli dosáhnout

## Inspirace

https://thispersondoesnotexist.com/

![thisperson](src/thisperson.jpg)

# Co je potřeba

::: incremental

* Vytvořit model, který by generoval obrázky dinosaurú
    * "State-of-the-art"
    * Prozkoumat a provnat různé modely
* Využití školního hpc clusteru pro trénovaní na grafických kartách
* Vytvoření webové aplikace

:::

# Sběr dat

Pro vytvoření trénovacích dat jsme použili https://www.kaggle.com/. Zejména pak:

* https://www.kaggle.com/datasets/larserikrisholm/dinosaur-image-dataset-15-species – 2448 obrázků - 510MB
* https://www.kaggle.com/datasets/cmglonly/simple-dinosurus-dataset - 200 obrázků - 58MB
* https://www.kaggle.com/datasets/antaresl/jurassic-park-dinosaurs-dataset - 4364 obrázků - 4GB 
* https://www.kaggle.com/datasets/caokhoihuynh/jurassic-world-dinosaur - 5835 obrázků - 1.6GB

# Pre-processing

* Obrázky jsou přeškálované na velikost 256x256 px
* Vzhledem k velikosti datasetu (celkem 12848 obrázků), jsme se rozhodli nezapojit žádné augmentační techniky

# Výběr modelu

## Stable Diffusion

* Iterativní odstraňování šumu
* Pro naše užití příliš výpočetně náročný

## DC-GAN

* Model má dvě části: Generátor a dikriminátor
* Přiliš těžká úloha, diskriminátor se vždy dosáhne úspěšnosti 100%

## VAE

* Enkóder -> Latentní prostor -> Dekóder
* Výpočetně nenáročný

# Trénování

![z_dim](src/zdim_loss.svg){ width=11cm } 

# Výsledky 

- 48 epoch

| 150 | 50 | 10 | 5 | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_0_0.png) | ![](./src/gen_images/vae/vae_50_0_13.png) | ![](./src/gen_images/vae/vae_10_0_1.png) | ![](./src/gen_images/vae/vae_5_0_43.png) | ![](./src/gen_images/vae/vae_2_0_6.png) |
| ![](./src/gen_images/vae/vae_150_0_7.png) | ![](./src/gen_images/vae/vae_50_0_44.png) | ![](./src/gen_images/vae/vae_10_0_45.png) | ![](./src/gen_images/vae/vae_5_0_55.png) | ![](./src/gen_images/vae/vae_2_0_20.png) |
| ![](./src/gen_images/vae/vae_150_0_30.png) | ![](./src/gen_images/vae/vae_50_0_36.png) | ![](./src/gen_images/vae/vae_10_0_58.png) | ![](./src/gen_images/vae/vae_5_0_69.png) | ![](./src/gen_images/vae/vae_2_0_27.png) |

# Výsledky 

- 96 epoch

| 150 | 50 | 10 | 5 | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_1_0.png) | ![](./src/gen_images/vae/vae_50_1_13.png) | ![](./src/gen_images/vae/vae_10_1_1.png) | ![](./src/gen_images/vae/vae_5_1_43.png) | ![](./src/gen_images/vae/vae_2_1_6.png) |
| ![](./src/gen_images/vae/vae_150_1_7.png) | ![](./src/gen_images/vae/vae_50_1_44.png) | ![](./src/gen_images/vae/vae_10_1_45.png) | ![](./src/gen_images/vae/vae_5_1_55.png) | ![](./src/gen_images/vae/vae_2_1_20.png) |
| ![](./src/gen_images/vae/vae_150_1_30.png) | ![](./src/gen_images/vae/vae_50_1_36.png) | ![](./src/gen_images/vae/vae_10_1_58.png) | ![](./src/gen_images/vae/vae_5_1_69.png) | ![](./src/gen_images/vae/vae_2_1_27.png) |

# Výsledky 

- 144 epoch

| 150 | 50 | 10 | 5 | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_2_0.png) | ![](./src/gen_images/vae/vae_50_2_13.png) | ![](./src/gen_images/vae/vae_10_2_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_2_6.png) |
| ![](./src/gen_images/vae/vae_150_2_7.png) | ![](./src/gen_images/vae/vae_50_2_44.png) | ![](./src/gen_images/vae/vae_10_2_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_2_20.png) |
| ![](./src/gen_images/vae/vae_150_2_30.png) | ![](./src/gen_images/vae/vae_50_2_36.png) | ![](./src/gen_images/vae/vae_10_2_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_2_27.png) |

# Výsledky 

- 192 epoch

| 150 | 50 | 10 | 5 (144 ep.) | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_3_0.png) | ![](./src/gen_images/vae/vae_50_3_13.png) | ![](./src/gen_images/vae/vae_10_3_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_3_6.png) |
| ![](./src/gen_images/vae/vae_150_3_7.png) | ![](./src/gen_images/vae/vae_50_3_44.png) | ![](./src/gen_images/vae/vae_10_3_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_3_20.png) |
| ![](./src/gen_images/vae/vae_150_3_30.png) | ![](./src/gen_images/vae/vae_50_3_36.png) | ![](./src/gen_images/vae/vae_10_3_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_3_27.png) |

# Výsledky 

- 240 epoch

| 150 | 50 | 10 | 5 (144 ep.) | 2 |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_4_0.png) | ![](./src/gen_images/vae/vae_50_4_13.png) | ![](./src/gen_images/vae/vae_10_4_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_4_6.png) |
| ![](./src/gen_images/vae/vae_150_4_7.png) | ![](./src/gen_images/vae/vae_50_4_44.png) | ![](./src/gen_images/vae/vae_10_4_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_4_20.png) |
| ![](./src/gen_images/vae/vae_150_4_30.png) | ![](./src/gen_images/vae/vae_50_4_36.png) | ![](./src/gen_images/vae/vae_10_4_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_4_27.png) |

# Výsledky 

- 288 epoch

| 150 | 50 | 10 | 5 (144 ep.) | 2 (240 ep.) |
|--|--|--|--|--|
| ![](./src/gen_images/vae/vae_150_5_0.png) | ![](./src/gen_images/vae/vae_50_5_13.png) | ![](./src/gen_images/vae/vae_10_5_1.png) | ![](./src/gen_images/vae/vae_5_2_43.png) | ![](./src/gen_images/vae/vae_2_4_6.png) |
| ![](./src/gen_images/vae/vae_150_5_7.png) | ![](./src/gen_images/vae/vae_50_5_44.png) | ![](./src/gen_images/vae/vae_10_5_45.png) | ![](./src/gen_images/vae/vae_5_2_55.png) | ![](./src/gen_images/vae/vae_2_4_20.png) |
| ![](./src/gen_images/vae/vae_150_5_30.png) | ![](./src/gen_images/vae/vae_50_5_36.png) | ![](./src/gen_images/vae/vae_10_5_58.png) | ![](./src/gen_images/vae/vae_5_2_69.png) | ![](./src/gen_images/vae/vae_2_4_27.png) |

# Výsledky

![](./latent_grid.png){ height=8cm }

# Diskuze

* Co jsme mohli ještě zkusit? (více modelů, větší VAE)
* Future work?



