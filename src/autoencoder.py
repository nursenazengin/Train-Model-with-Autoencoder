# src/train.py
import os
import numpy as np
import tensorflow as tf
from preprocess import DataLoader, DenseAutoencoder
from sklearn.model_selection import train_test_split

base = "/content/drive/MyDrive/unsupervised_auto_encoder"
train_path = f"{base}/data/train.csv"
model_weights_path = f"{base}/models/model.weights.h5"

loader = DataLoader()
x_full, _ = loader.load_train_data(train_path)

seed=42
np.random.seed(seed)
tf.random.set_seed(seed)
x_train, x_val = train_test_split(x_full, test_size=0.4, random_state=seed)

autoencoder = DenseAutoencoder(input_dim=x_train.shape[1])
autoencoder.train(x_train, x_val)
autoencoder.save(model_weights_path)

print("✅ Model eğitildi ve kaydedildi.")
