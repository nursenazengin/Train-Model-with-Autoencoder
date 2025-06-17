# src/preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras._tf_keras.keras.layers import Input, Dense
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import joblib 

class DataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def load_train_data(self, path):
        df = pd.read_csv(path)
        if('0' in df.columns):
          x = df.drop(columns=['0']).values
          y = df['0'].values
        else:
          x = df.drop(columns=['1']).values
          y = df['1'].values
          
        x_scaled = self.scaler.fit_transform(x)

        joblib.dump(self.scaler, "/content/drive/MyDrive/unsupervised_auto_encoder/models/scaler.pkl")
        return x_scaled, y

    def load_predict_data(self, path):
        df = pd.read_csv(path)
        self.scaler = joblib.load("/content/drive/MyDrive/unsupervised_auto_encoder/models/scaler.pkl")
        x_scaled = self.scaler.transform(df.values)
        return x_scaled, df

    def load_actual_labels(self, path):
        df = pd.read_csv(path)
        return df['0'].values

class DenseAutoencoder:
    def __init__(self, input_dim, latent_dim=3):
        self.model = self._build(input_dim, latent_dim)
        self.model.compile(optimizer=Adam(0.001), loss='mse')

    def _build(self, input_dim, latent_dim):
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        latent = Dense(latent_dim, activation='relu')(x)
        x = Dense(32, activation='relu')(latent)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(input_dim, activation='sigmoid')(x)
        return Model(inputs, outputs)

    def train(self, x_train, x_val, epochs=70, batch_size=64):
        self.model.fit(x_train, x_train, validation_data=(x_val,x_val),
                       epochs=epochs, batch_size=batch_size, shuffle=True)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            self.model.load_weights(path)
        else:
            raise FileNotFoundError("Model weights not found.")

class AnomalyDetector:
    def __init__(self, model, threshold=0.02):
        self.model = model
        self.threshold = threshold

    def predict(self, x):
        x_pred = self.model.predict(x)
        mse = np.mean(np.power(x - x_pred, 2), axis=1)
        y_pred = (mse > self.threshold).astype(int)
        return y_pred
