
# INF431 - Projet 9 : GDPR in practice



## Intégration Python - C++

### Généralités

Notre projet présente une architecture hybride :

- un _back-end_ C++, partie du programme qui fait le gros des calculs et de la gestion de la mémoire.
- un _front-end_ Python qui présente certaines fonctions et certains objets de la librairie C++ sous la forme de fonctions et objets Python dans une librairie appelée `info9`.

Cette architecture permet d'utiliser les fonctions que nous avons implémentées en C++ dans un _Jupyter Notebook_, par exemple. Le _back-end_ C++ peut également être compilé en une librairie C++ traditionnelle.

### En détail

Cette architecture est rendue possible par l'utilisation de la librairie [`pybind11`](https://pybind11.readthedocs.io/en/stable/intro.html) : elle permet de créer des _bindings_ Python pour une librairie C++ existante. Les _bindings_ ainsi générés prennent la forme d'une librairie qui peut être importée par un script Python de manière transparente (i.e. le script Python n'a aucune connaissance de la librarie `pybind11` ; il contient simplement la ligne `import info9`).

La librairie C++ (en dehors d'un fichier `wrap.cpp` qui permet de générer les bindings) n'a elle-même aucune connaissance de la librairie `pybind11` : elle peut être compilée et utilisée par un programme C++ standard.

La grande force de `pybind11` est donc de découpler très largement la partie C++ de la partie Python, tout en permettant des appels de fonction C++ par Python de façon peu coûteuse. Notre projet utilise notamment un mécanisme très intéressant de `pybind11` qui consiste à unifier deux types :

- les `numpy.ndarray` de dimension 2 de la librairie `numpy` du côté Python
- les `Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>` de la librairie `Eigen` du côté C++, qui correspondent à une version "row-major" des `Eigen::MatrixXd`, qui sont elles stockées en "column-major". Ce type a été renommé `RMatrixXd` dans notre code.

Dans les faits, il est possible de faire transiter ces types entre la librairie C++ et le script Python **sans copie** ! De même, les `numpy.ndarray` de dimension 1 et les `Eigen::VectorXd` sont unifiés.

`pybind11` permet également de gérer facilement la durée de vie des objets partagés (ou déplacés) entre C++ et Python en faisant coopérer les `std::shared_ptr` avec le _refcounting_ de Python.

<div class="box">
La fonction suivante définie dans le code C++ :

```cpp
RMatrixXd add(Eigen::Ref<RMatrixXd const> const &a, Eigen::Ref<RMatrixXd const> const &b) {
  return a + b;
}
```
pourra être appelée par un script Python sans copie des arguments :

```python3
import info9
import numpy as np
A = np.array([1.,2.,3.])
B = info9.add(a, a) # zero-copy!
```


Avec ces appels sans copie, `pybind11` permet également les fonctions en-place. La fonction

```cpp
void transpose(Eigen::Ref<RMatrixXd> matrice_a_modifier) {
  a.transposeInPlace();
}
```
permettra de transposer la matrice `A`:

```python3
info9.transpose(a)
```
</div>


<div class="box">
Note d'implémentation : `Eigen::Ref` est la méthode traditionnelle utilisée pour passer des objets Eigen à une fonction sans faire de copie. Utiliser une référence `Eigen::MatrixXd &` ne fonctionnerait pas aussi bien : les opérations même simples entre deux matrices Eigen (`a * b` par exemple) ne renvoient pas des matrices Eigen, mais des objets qui seront évalués de manière paresseuse.  Attention cependant : les `Eigen::Ref` peuvent référencer de la mémoire qui a été déallouée (par exemple si une `Eigen::Ref` pointe vers une variable automatique et est renvoyée par une fonction) !
</div>



## Problème 1 : Classifieur supervisé

Le jeu de données en entrée correspond à un ensemble de points labelisés dans un espace de dimension 1024. Chaque point correspond à un token dans un corpus de phrases en anglais ; le label donne des informations sur la sémantique de ce token dans la phrase (i.e. si c'est un lieu par "I-LOC", une personne par "I-PER", ...). Le but de cette première tâche est, étant donnée un deuxième ensemble de tokens non-labellisés, de prédire si ces tokens correspondent à des personnes.

Cette tâche correspond donc à processus d'apprentissage supervisé : on va entraîner un classifieur sur le premier jeu de données, et l'utiliser sur le deuxième jeu de données pour y prédire les labels.


### Notes sur le stockage des jeux de données

Les jeux de données ont été convertis au format [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format). Ce format de fichier binaire est très bien supporté depuis Python par la librairie [h5py](https://www.h5py.org/) qui permet d'y lire des données sous la forme de `numpy.ndarray`, et en C++ par la librairie [HighFive](https://bluebrain.github.io/HighFive/), qui permet d'y lire des données sous la forme d'objets `Eigen`.


### Classifieur k-NN

Nous avons commencé par implémenter le classifieur k-NN. -> implémente réduction de la dimension par "random-projection".

-> choisir une vision, ou détailler les deux
1. classifier binaire (I-PERS vs autre)
2. classifier à n labels : et alors vote majoritaire, ou threshold pour les labels != O ? A tester !

### Notes sur l'implémentation

Notre implémentation reprends le code source donné en TD6. Elle utilise les librairies suivantes:
- Eigen pour le stockage des vecteurs et les calculs matriciels.
- [ANN](http://www.cs.umd.edu/~mount/ANN/) pour la recherche de plus proches voisins utilisant un kd-tree.
- [HighFive](https://bluebrain.github.io/HighFive/) pour lire les fichiers HDF5
- [OpenMP](https://www.openmp.org/), indirectement, pour paralléliser les calculs de la librairie Eigen. Il suffit de _link_er cette librairie pour qu'Eigen parallélise certaines opérations.

Des modifications importantes ont cependant été apportées au code source donné au TD6:
- le jeu de données n'est plus stocké en tant que `std::vector<std::vector<double>>` mais en tant que `RMatrixXd`. Chaque ligne de cette matrice est un vecteur dont on peut obtenir une référence sans copie avec la méthode `RMatrixXd::row(int i)`. Les labels sont stockés à part dans un vecteur.
- nous n'utilisons pas de `ANNpointArray` ou de `ANNpoint`, mais passons directement à la librairie ANN des pointeurs vers les vecteurs du jeu de données. Notamment toutes les opérations de prédiction (dans la fonction `KnnClassifier::Estimate`) sont menées sans effectuer de copie de vecteurs.
- le jeu de donnés étant stocké sous la forme d'une matrice, l'opération de projection dans un espace de plus petite dimension se rammène à une simple multiplication matricielle parallélisée par la librairie `Eigen`.

### Analyse

- même pour des $k$ relativement faible ($k = 10$) et une forte réduction de la dimension (de $d = 1024$ à $l = 64$), le classifier k-NN reste particulièrement lent.
- énorme perte de performance après la projection <- une erreur dans l'implémentation ?
