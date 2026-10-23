# Reconnaissance de panneaux de signalisation avec un CNN

Projet pédagogique en français pour classifier un panneau routier parmi **43 classes** avec TensorFlow/Keras et le dataset **GTSRB** de Kaggle.

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yassinelaouni/gtsrb-traffic-sign-recognition/blob/main/traffic_sign_recognition.ipynb)

📄 [Lire le rapport PDF illustré](output/pdf/gtsrb_explication_projet.pdf) : 8 pages en français pour comprendre les données, le CNN, l'entraînement, l'évaluation et l'utilisation dans Colab. Les illustrations générées avec ChatGPT et leurs prompts sont conservés dans [le dossier des illustrations](output/pdf/assets/README.md).

Le bouton Colab fonctionnera après le premier push du notebook sur GitHub. Pour démarrer immédiatement, importer `traffic_sign_recognition.ipynb` dans [Google Colab](https://colab.research.google.com/) via **Fichier → Importer un notebook**.

## Utiliser le notebook

1. Importer le notebook dans Colab.
2. Choisir **Exécution → Modifier le type d'exécution → GPU**, si disponible.
3. Exécuter les cellules dans l'ordre. Le dataset est téléchargé automatiquement avec KaggleHub.
4. Consulter les courbes, l'accuracy, le F1 macro, la matrice de confusion et les erreurs.
5. Activer `DOWNLOAD_RESULTS = True` pour télécharger le modèle et les résultats.
6. Pour essayer une photo, définir `RUN_UPLOAD_DEMO = True` dans la configuration puis exécuter la cellule de prédiction. Recadrer l'image sur un seul panneau.

`SMOKE_TEST = True` permet une vérification rapide avec une époque et quelques images par classe. Ce mode ne mesure pas les performances réelles du modèle. L'entraînement complet est activé par défaut et peut demander du temps ; aucun score final n'est annoncé à l'avance.

## Dataset

- [GTSRB — German Traffic Sign Recognition Benchmark sur Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), version 1.
- 39 209 images d'entraînement et 12 630 de test, soit 51 839 images, réparties en 43 classes.
- Les CSV `Train.csv` et `Test.csv` fournissent les chemins et les classes ; le notebook contrôle leurs effectifs, les chemins et la présence des classes.
- Validation : premier pli de `StratifiedGroupKFold` à cinq plis, avec les vues d'une même séquence regroupées. Cela limite la fuite entre images voisines d'un même panneau.
- Le test officiel reste séparé pendant l'entraînement et la sélection du meilleur checkpoint.

Le dataset public est généralement accessible sans clé. Si Kaggle exige une authentification, exécuter `kagglehub.login()` séparément et relancer le téléchargement. Ne jamais publier de clé. Le notebook accepte aussi `LOCAL_DATA_DIR` pour un dataset déjà extrait.

## Méthode

Images RGB redimensionnées à 48 × 48 ; normalisation incluse dans le modèle ; augmentation légère sans retournement ; trois blocs de convolutions avec batch normalization, pooling et dropout ; classification softmax à 43 sorties. Adam, arrêt anticipé, réduction du taux d'apprentissage et sauvegarde du meilleur modèle selon la perte de validation.

Les résultats exportés dans `artifacts/full/` comprennent le modèle `.keras`, les noms des classes, les métriques, les courbes, les prédictions du test, le rapport par classe, la matrice de confusion, les versions logicielles et le manifeste des splits. `artifacts/gtsrb_results.zip` les regroupe pour téléchargement. Le mode rapide utilise des chemins distincts.

## Vérification effectuée

Le schéma Jupyter et la syntaxe Python ont été validés. Les cellules ont été exécutées en mode rapide sur de vraies images Kaggle : 344 images d'apprentissage, 172 de validation et 172 de test, sur une époque avec TensorFlow CPU 2.21 et Python 3.12. L'évaluation, la prédiction, l'export `.keras`, le rechargement avec comparaison des sorties et l'intégrité de l'archive ont réussi. L'entraînement complet sur GPU dans Colab reste à exécuter ; les scores du mode rapide ne constituent pas un résultat benchmark.

## Exécution locale

Python 3.11 ou 3.12 est recommandé. Dans un environnement virtuel :

```bash
python -m pip install -r requirements.txt
jupyter lab traffic_sign_recognition.ipynb
```

Colab utilise son propre TensorFlow ; la cellule d'installation du notebook ne le remplace pas. Les plages de versions du fichier de dépendances ne constituent pas un verrouillage exact ; les versions réellement utilisées sont enregistrées à l'export.

## Publier sur GitHub

Depuis le dossier de ce projet, après vérification de `git status` :

```bash
git add README.md traffic_sign_recognition.ipynb requirements.txt LICENSE .gitignore
git commit -m "Add GTSRB CNN notebook for Google Colab"
git push -u origin main
```

Le dépôt distant prévu est [yassinelaouni/gtsrb-traffic-sign-recognition](https://github.com/yassinelaouni/gtsrb-traffic-sign-recognition). Les données, modèles, environnements et secrets sont exclus par `.gitignore`.

## Limites

Ce projet classe un panneau déjà cadré ; il ne détecte pas plusieurs panneaux dans une scène. Le modèle est limité aux classes GTSRB et peut se tromper sur des panneaux d'autres pays, une image floue ou un objet inconnu. Un score softmax élevé ne garantit pas une bonne prédiction. Le projet est pédagogique et ne convient pas à la prise de décisions de conduite.

## Sources et licence

- [Benchmark original GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html).
- [KaggleHub officiel](https://github.com/Kaggle/kagglehub).
- [TensorFlow : augmentation des données](https://www.tensorflow.org/tutorials/images/data_augmentation).
- [Scikit-learn : StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

Le code est sous licence MIT. Les données ne sont pas incluses : consulter les conditions et les références du dataset Kaggle et du benchmark original.
