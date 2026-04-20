# ======================================================================
# Réseau de neurones sur IRIS avec TensorFlow / Keras
# - Chargement et préparation des données
# - Construction du modèle (réseau de neurones dense)
# - Entraînement + courbes d'apprentissage
# - Matrice de confusion
# - Courbes ROC multi-classe
# ======================================================================

import numpy as np
import matplotlib.pyplot as plt

from typing import cast

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical


# ----------------------------------------------------------------------
# 1. Chargement du dataset IRIS
# ----------------------------------------------------------------------
# Utilisation d'un accès par clés pour éviter les faux positifs de Pylance
# sur les attributs .data, .target et .target_names
iris = load_iris()  # 150 échantillons, 4 caractéristiques, 3 classes
X = np.asarray(iris["data"])  # shape (150, 4)
y = np.asarray(iris["target"])  # labels : 0 = setosa, 1 = versicolor, 2 = virginica
class_names = np.asarray(iris["target_names"])

# ----------------------------------------------------------------------
# 2. Normalisation des données (centrage-réduction)
# ----------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------
# 3. Encodage des labels en one-hot pour Keras
# ----------------------------------------------------------------------
y_cat = to_categorical(y, num_classes=3)

# ----------------------------------------------------------------------
# 4. Découpage entraînement / test
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------------------
# 5. Construction du réseau de neurones
# Réseau dense à 2 couches cachées, comme décrit dans le RI
# ----------------------------------------------------------------------
model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

# ----------------------------------------------------------------------
# 6. Compilation du modèle
# - Fonction de perte : categorical_crossentropy (classification multi-classe)
# - Optimiseur : Adam (descente de gradient améliorée)
# - Métrique : accuracy
# ----------------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ----------------------------------------------------------------------
# 7. Entraînement du modèle
# ----------------------------------------------------------------------
# On retire verbose=0 car certaines versions/stubs Keras déclenchent
# un faux positif Pylance sur ce paramètre
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=8
)

# ----------------------------------------------------------------------
# 8. Évaluation finale sur l'ensemble de test
# ----------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Loss sur le test : {test_loss:.4f}")
print(f"Accuracy sur le test : {test_acc * 100:.2f}%")

# ----------------------------------------------------------------------
# 9. Courbes d'apprentissage (loss et accuracy)
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 4))

# Courbe de loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="loss entraînement")
plt.plot(history.history['val_loss'], label="loss validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Évolution de la fonction de perte")
plt.legend()

# Courbe d'accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="accuracy entraînement")
plt.plot(history.history['val_accuracy'], label="accuracy validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Évolution de la précision")
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 10. Matrice de confusion
# ----------------------------------------------------------------------
# Prédictions sur l'ensemble de test
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# Attention à ne pas appeler la matrice "cm", sinon cela masque
# un éventuel alias matplotlib.cm
conf_mat = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Matrice de confusion - IRIS")
plt.show()

# ----------------------------------------------------------------------
# 11. Courbes ROC multi-classe
# ----------------------------------------------------------------------
# Binarisation des labels pour calculer une ROC par classe
# Conversion explicite en ndarray pour éviter le faux positif Pylance
# sur le type spmatrix lors de l'indexation y_test_bin[:, i]
y_test_bin = cast(np.ndarray, np.asarray(label_binarize(y_true, classes=[0, 1, 2])))
n_classes = y_test_bin.shape[1]

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Tracé des courbes ROC
plt.figure(figsize=(6, 5))
colors = ['darkorange', 'green', 'blue']

for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f"Classe {class_names[i]} (AUC = {roc_auc[i]:0.2f})"
    )

plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Aléatoire")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Courbes ROC multi-classe - IRIS")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()