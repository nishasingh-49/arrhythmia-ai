import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
print("Current working directory:", os.getcwd())
print("Loading dataset...")
train_df = pd.read_csv("archive/mitbih_train.csv", header=None)
test_df = pd.read_csv("archive/mitbih_test.csv", header=None)
df = pd.concat([train_df, test_df], axis=0)
print("Total samples:", len(df))
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = np.where(y == 0, 0, 1)
print("Normal:", np.sum(y == 0))
print("Abnormal:", np.sum(y == 1))
print("Abnormal ratio:", np.sum(y == 1)/len(y))
#train test split satisfied.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train size:", len(X_train))
print("Test size:", len(X_test))
#normalization.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
import joblib
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved successfully!")
X_test = scaler.transform(X_test)
#reshapind for cnn
X_train = X_train.reshape(-1, X_train.shape[1], 1)
X_test = X_test.reshape(-1, X_test.shape[1], 1)
#for handling class imbalance.
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weights)
#1D cnn.
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=(X_train.shape[1],1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(128, 3, activation='relu'),
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
#train.
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)
#evaluate.
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_pred_prob))
#save model.
model.save("arrhythmia_model.keras")
print("\nModel saved successfully!")


