# src/predict.py
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from preprocess import DataLoader, DenseAutoencoder, AnomalyDetector
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

base = "/content/drive/MyDrive/unsupervised_auto_encoder"
predict_path = f"{base}/data/predict.csv"
prediction_result_path = f"{base}/data/prediction_results.csv"
failure_predict_path = f"{base}/data/failure_predict.csv"
model_weights_path = f"{base}/models/model.weights.h5"

loader = DataLoader()
x_predict, predict_df = loader.load_predict_data(predict_path)
y_actual = loader.load_actual_labels(failure_predict_path)

autoencoder = DenseAutoencoder(input_dim=x_predict.shape[1])
autoencoder.load(model_weights_path)

detector = AnomalyDetector(autoencoder.model)
y_pred = detector.predict(x_predict)

# Tahminleri kaydet
pd.DataFrame({'anomaly': y_pred}).to_csv(prediction_result_path, index=False)

if len(y_pred) != len(y_actual):
  min_length = min(len(y_pred), len(y_actual))
  y_pred = y_pred[:min_length]
  y_actual = y_actual[:min_length]

# Değerlendirme
acc = accuracy_score(y_actual, y_pred)
prec = precision_score(y_actual, y_pred)
rec = recall_score(y_actual, y_pred)
f1 = f1_score(y_actual, y_pred)
cm = confusion_matrix(y_actual, y_pred)


print("Accuracy :", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("F1 Score :", round(f1, 4))
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
#plt.show()
display(plt.gcf())


metrics = [acc, prec, rec, f1]
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.figure(figsize=(6, 4))
bars=plt.bar(x=labels, height=metrics, color=sns.color_palette("pastel"))
plt.ylim(0, 1)
plt.title("Model Performance Metrics")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.2f}", ha="center", va='bottom')

plt.tight_layout()
#plt.show()
display(plt.gcf())






      
