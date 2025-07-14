# 🧠 Moroccan City Name Generator with Transformers + SAE

Ce projet explore l'utilisation des **modèles Transformers** pour générer des noms de villes marocaines de manière cohérente et linguistiquement plausible. Il va plus loin en intégrant un **Système d'Aide à l'Explication (SAE)** permettant d'interpréter les activations internes du modèle, et même de **modifier son comportement** en temps réel.

---

## 📁 Structure du projet

```
.
├── datapreparation.py     # Préparation des données et analyse initiale
├── model.py               # Définition du modèle Transformer + génération + analyse
├── interpretabilite.py    # SAE : Interprétation des activations internes
├── comportement.py        # Modification du comportement du modèle via les activations
├── weights/               # Dossier contenant les poids entraînés du modèle
└── README.md              # Ce fichier
```

---

## 🔍 Objectif

- Générer des noms de villes marocaines à l’aide d’un Transformer entraîné sur un corpus réel.
- Analyser les **structures linguistiques internes** apprises par le modèle.
- Construire un **SAE** pour comprendre les activations internes.
- Manipuler les **activations neuronales** pour modifier le comportement du modèle (ex. forcer l’apparition de certains motifs).

---

## 📦 Dépendances

Assure-toi d’avoir Python 3.8+ et installe les dépendances via :

```bash
pip install torch numpy matplotlib seaborn scikit-learn
```

---

## 📊 Étapes du projet

### 1. 🔧 `datapreparation.py`

- Nettoie et normalise les noms de villes marocaines.
- Crée un tokenizer caractère par caractère.
- Analyse linguistique :
  - **Fréquence du nombre de composants** (préfixes/suffixes).
  - **Nombre de composants par nom**.

➡️ Résultat : corpus prêt + vocabulaire + graphiques analytiques.

---

### 2. 🧠 `model.py`

- Implémente un modèle **Transformer** minimal pour la génération.
- Entraîne le modèle sur les données préparées.
- Génère de nouveaux noms de villes marocaines.
- Analyse de sortie :
  - Répartition des composants générés.
  - Comparaison statistique avec les données d’entraînement.
- 📌 **Les poids du modèle sont automatiquement sauvegardés dans le dossier `weights/` à l'issue de l'entraînement.**

➡️ Résultat : modèle entraîné + noms générés + analyse comparative.

---

### 3. 🪄 `interpretabilite.py`

- Implémente un **SAE (Système d’Aide à l’Explication)**.
- Capture les **activations** des neurones sur le corpus d'entraînement.
- Regroupe les activations par **features latentes** (clustering).
- Analyse de **séquences d'activations** :
  - Mise en évidence de motifs syntaxiques ou phonétiques appris.

➡️ Résultat : concepts internes identifiés, graphiques d’interprétation.

---

### 4. 🎛️ `comportement.py`

- Modifie dynamiquement le **comportement du modèle** en agissant sur les activations internes.
- Expérimentations :
  - Forcer certains patterns (préfixes amazigh, arabes, etc.).
  - Supprimer/inhiber certains concepts latents.
- Analyse des sorties générées après modification.

➡️ Résultat : démonstration du **contrôle conceptuel** sur la génération.


## 🧠 Inspirations et Références

- (https://www.youtube.com/watch?v=n4EnafoZ38Q&ab_channel=AlexandreTL)
- https://transformer-circuits.pub/2023/monosemantic-features/index.html
- https://youtu.be/jGCvY4gNnA8?si=Aw8Mhe3_iQEqSBkk

---

## 📜 Licence

Projet open-source sous licence MIT.

---

## ✍️ Auteur

**Reda Hamama**  
