import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ---------------------------------------------------
# 1. LOAD YOUR DATA
# ---------------------------------------------------

# Replace these with your actual variables
# beats = np.array([...])  # shape (N, 200)
# labels = np.array([...]) # shape (N,)

# If you're loading from file:
data = np.load("processed_beats.npz")
beats = data["beats"]
labels = data["labels"]

print("Total samples:", len(labels))
print("Normal:", np.sum(labels == 0))
print("Abnormal:", np.sum(labels == 1))
print("Abnormal ratio:", np.sum(labels == 1) / len(labels))


# ---------------------------------------------------
# 2. TRAIN / TEST SPLIT (STRATIFIED)
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    beats,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# ---------------------------------------------------
# 3. NORMALIZATION
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------
# 4. RESHAPE FOR CNN (samples, timesteps, channels)
# ---------------------------------------------------

X_train = X_train.reshape(-1, 200, 1)
X_test = X_test.reshape(-1, 200, 1)

print("New shape:", X_train.shape)


# ---------------------------------------------------
# 5. HANDLE CLASS IMBALANCE
# ---------------------------------------------------

weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

print("Class weights:", class_weights)


# ---------------------------------------------------
# 6. BUILD 1D CNN MODEL
# ---------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=(200,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()


# ---------------------------------------------------
# 7. TRAIN MODEL
# ---------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)


# ---------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))


# ---------------------------------------------------
# 9. SAVE MODEL
# ---------------------------------------------------

model.save("arrhythmia_model.h5")
print("Model saved as arrhythmia_model.h5")
