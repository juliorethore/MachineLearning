import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# 1. Chargement du dataset Iris
iris = load_iris()  # 150 échantillons, 4 features, 3 classes
X = iris.data       # (150, 4)
y = iris.target     # labels 0,1,2

# 2. Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. One-hot encoding des labels
y_cat = to_categorical(y, num_classes=3)

# 4. Découpage entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y
)

# 5. Construction du réseau de neurones
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))   # 1ère couche cachée
model.add(Dense(8, activation='relu'))                      # 2e couche cachée
model.add(Dense(3, activation='softmax'))                   # couche de sortie (3 classes)

# 6. Compilation du modèle
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Entraînement
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=8,
    verbose=0
)

# 8. Évaluation finale
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy sur le test : {test_acc*100:.2f}%")

# 9. Courbes d’apprentissage (loss et accuracy)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="loss entraînement")
plt.plot(history.history['val_loss'], label="loss validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="acc entraînement")
plt.plot(history.history['val_accuracy'], label="acc validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
