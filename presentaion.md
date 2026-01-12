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

# Diskuze

* Co jsme mohli ještě zkusit? (více modelů, větší VAE)
* Future work?

# 


