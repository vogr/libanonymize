
# INF442 - Projet 9 : GDPR in practice

## Introduction

Ce projet implémente différentes méthodes de classification utilisée pour anonymiser des données. Ce fichier `README.md` vous guidera pour l'installation et l'exécution de ce projet sur des exemples.

Notre projet présente une architecture hybride :

- un _back-end_, partie du programme qui fait le gros des calculs et de la gestion de la mémoire, implémentée dans une librairie C++ appelée `libinfo9`.
- un _front-end_ Python qui présente certaines fonctions et certains objets de la librairie `libinfo9` sous la forme de fonctions et objets Python dans une librairie Python appelée `info9`.

Cette architecture permet d'utiliser les fonctions que nous avons implémentées en C++ dans un _Jupyter Notebook_, par exemple. Le _back-end_ C++ peut également être appelé depuis des programmes C++ traditionnels.

## Description des dossiers du projet

Le projet s'organise de la manière suivante :

- `conda_env` : dossier contenant les fichiers de configuration pour mettre en place un environnement Conda.
- `data` :  des jeux de données ConLL (1 pour le *training*, deux pour le *testing*) sous leur forme originale (`.conll`), et après leur transformation par [BERT](https://github.com/google-research/bert) stockés au format binaire [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format).
- `examples_cpp`: un ensemble de programmes compilables utilisant la librairie `libinfo9` depuis C++.
- `notebooks`: un ensemble de Jupyter Notebooks utilisant de la librairie `info9` depuis Python.
- `pkg_info9`: l'implémentation en C++ de la librairie `libinfo9`

## Installation

### Prérequis

Toutes les librairies (C++ et Python) seront installées dans un environnement Conda. Il est donc nécessaire d'installer [miniconda](https://docs.conda.io/en/latest/miniconda.html) ; des builds pour Windows, MacOS et Linux sont disponibles [sur la page dédiée](https://docs.conda.io/en/latest/miniconda.html).

### Mise en place de l'environnement Conda et installation

Les commandes suivantes permettent de mettre en place l'environnement Conda et d'y installer la librairie `info9`:

```console
# A exécuter depuis le dossier principal du projet :

# Création de l'environnement et installation des paquets
$ conda env update --name "info_projet9" --file "./conda_env/conda_env.yml"

# Activation de l'environnement
$ conda activate "info_projet9"

# Installation du paquet info9 dans l'environnement conda
(info_projet9) $ python -m pip install "../pkg_info9"
```

## Utilisation des exemples

### Exemples Python

Les _Jupyter Notebooks_ du dossier `notebooks/` présente l'utilisation de la librairie `info9`. Nous y présentons également des analyses des différents classifieurs, et explorons des méthodes de prétraitement des jeux de données.

__Attention :__ Ces _Jupyter Notebooks_ doivent impérativement être ouverts dans l'environnement Conda pour avoir accès aux librairies nécessaires, et notamment à `info9` :

```console
(info_projet9) $ python -m jupyter notebook "./notebooks/"

# Ou de manière équivalente :
(info_projet9) $ ./notebooks/notebooks.sh
```

Pour que le notebook `Vectorisation.ipynb` fonctionne correctement, il faudra au préalable télécharger le dictionnaire de _word embedding_ Glove, par exemple avec le script
```
$ ./notebooks/glove_pretrained/get.sh
```

### Exemples C++

Les sous-dossiers de `examples_cpp` contiennent des programmes compilables C++ présentant ou testant un aspect de la librairie `libinfo9`. Ils suivent tous le même modèle:

- un script `build.sh` permettant de compiler le programme avec la librairie `libinfo9`. Ce script appelle simplement CMake.
- un script `run.sh` permettant d'exécuter le programme compilé sur des données, avec des paramètres par défauts choisis à l'avance.
- un fichier source `main.cpp`, où le programme est implémenté.

__Attention :__ Les programmes s'exécutent bien plus rapidement lorsqu'ils sont compilés avec les optimisations. C'est le mode par défaut utilisé par `build.sh`. De manière équivalente, on pourra donc utiliser:
```console
# Pour des builds optimisés
$ ./build.sh
# ou
$ ./build.sh Release
# ou
$ cmake -S . -B "./build" -GNinja -DCMAKE_BUILD_TYPE=Release
$ cmake --build "./build"
```
