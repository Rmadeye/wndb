import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd


class SpaceshipModel:
    def __init__(self):
        self.model = None
        self.history = None

    def build_model(
        self,
        actfn1="relu",
        actfn2="relu",
        dropout1=0.3,
        dropout2=0.3,
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    ):
        # Budowa modelu
        self.model = models.Sequential(
            [
                layers.InputLayer(input_shape=(19,)),
                layers.Dense(64, activation=actfn1),
                layers.Dropout(dropout1),
                layers.Dense(32, activation=actfn2),
                layers.Dropout(dropout2),
                layers.Dense(1, activation="sigmoid"),  # Binarny output
            ]
        )

        # Kompilacja modelu
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, X_train, y_train, X_val, y_val,
              epochs=10, batch_size=32, **kwargs):
        # Trening modelu
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

    def evaluate(self, X_test, y_test):
        # Ewaluacja modelu na zbiorze testowym
        loss, accuracy = self.model.evaluate(X_test, y_test)

        return loss, accuracy

    def predict(self, X):
        # Predykcje na nowym zbiorze danych
        return (self.model.predict(X) > 0.5).astype("int32")
