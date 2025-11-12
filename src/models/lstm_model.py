"""LSTM-based sign language recognition model."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .base_model import BaseSignModel


class LSTMSignModel(BaseSignModel):
    """LSTM model for sign language recognition."""
    
    def build(self, num_classes):
        """
        Build the LSTM model architecture.
        
        Args:
            num_classes: Number of sign classes to recognize
        """
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            
        Returns:
            history: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
