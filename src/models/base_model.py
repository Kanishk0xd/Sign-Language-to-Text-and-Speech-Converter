"""Base class for sign language recognition models."""

from abc import ABC, abstractmethod
import numpy as np


class BaseSignModel(ABC):
    """Abstract base class for sign recognition models."""
    
    def __init__(self, sequence_length=30, num_features=126):
        """
        Initialize the base model.
        
        Args:
            sequence_length: Number of frames in a sequence
            num_features: Number of features per frame (21 landmarks * 3 coords * 2 hands)
        """
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.classes = []
        
    @abstractmethod
    def build(self, num_classes):
        """
        Build the model architecture.
        
        Args:
            num_classes: Number of sign classes to recognize
        """
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """
        Train the model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
        """
        pass
    
    def predict(self, sequence):
        """
        Predict the sign from a sequence.
        
        Args:
            sequence: Array of shape (sequence_length, num_features)
            
        Returns:
            predicted_class: Index of predicted class
            confidence: Confidence score
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = np.expand_dims(sequence, axis=0)
        
        predictions = self.model.predict(sequence, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return predicted_class, confidence
    
    def save(self, path):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
    
    def set_classes(self, classes):
        """
        Set the class labels.
        
        Args:
            classes: List of class names
        """
        self.classes = classes
    
    def get_class_name(self, class_index):
        """
        Get the class name from index.
        
        Args:
            class_index: Index of the class
            
        Returns:
            class_name: Name of the class
        """
        if class_index < len(self.classes):
            return self.classes[class_index]
        return f"unknown_{class_index}"
