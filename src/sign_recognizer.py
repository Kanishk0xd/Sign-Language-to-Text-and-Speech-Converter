"""Real-time sign language recognition system."""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple

from .hand_tracker import HandTracker
from .models import BaseSignModel
from .utils import TTSEngine


class SignRecognizer:
    """Real-time sign language recognition with text and speech output."""
    
    def __init__(self, model: BaseSignModel, config: dict):
        """
        Initialize the sign recognizer.
        
        Args:
            model: Trained sign recognition model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Initialize hand tracker
        hands_config = config.get('hands', {})
        self.hand_tracker = HandTracker(
            max_num_hands=hands_config.get('max_num_hands', 2),
            min_detection_confidence=hands_config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=hands_config.get('min_tracking_confidence', 0.5)
        )
        
        # Initialize TTS if enabled
        tts_config = config.get('tts', {})
        self.tts_enabled = tts_config.get('enabled', True)
        if self.tts_enabled:
            self.tts = TTSEngine(
                mode=tts_config.get('mode', 'offline'),
                rate=tts_config.get('rate', 150),
                volume=tts_config.get('volume', 1.0),
                language=tts_config.get('language', 'en')
            )
        else:
            self.tts = None
        
        # Sequence buffer
        model_config = config.get('model', {})
        self.sequence_length = model_config.get('sequence_length', 30)
        self.threshold = model_config.get('threshold', 0.8)
        self.sequence = deque(maxlen=self.sequence_length)
        
        # Vocabulary mapping
        self.vocabulary = config.get('vocabulary', {})
        
        # Display settings
        self.display_config = config.get('display', {})
        
        # State tracking
        self.current_text = ""
        self.last_prediction = None
        self.frame_count = 0
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            frame: Processed frame with annotations
            prediction: Predicted sign text (if confident enough)
        """
        # Extract hand landmarks
        landmarks, results = self.hand_tracker.extract_landmarks(frame)
        
        # Add to sequence buffer
        self.sequence.append(landmarks)
        
        # Draw landmarks if enabled
        if self.display_config.get('show_landmarks', True):
            frame = self.hand_tracker.draw_landmarks(frame, results)
        
        prediction_text = None
        
        # Make prediction when sequence is full
        if len(self.sequence) == self.sequence_length:
            sequence_array = np.array(list(self.sequence))
            
            try:
                class_idx, confidence = self.model.predict(sequence_array)
                
                if confidence >= self.threshold:
                    class_name = self.model.get_class_name(class_idx)
                    
                    # Map to vocabulary phrase if available
                    prediction_text = self.vocabulary.get(class_name, class_name)
                    
                    # Only update if prediction changed
                    if prediction_text != self.last_prediction:
                        self.current_text = prediction_text
                        self.last_prediction = prediction_text
                        
                        # Trigger TTS
                        if self.tts_enabled and self.tts:
                            self.tts.speak(prediction_text)
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Draw caption
        self._draw_caption(frame, self.current_text)
        
        return frame, prediction_text
    
    def _draw_caption(self, frame: np.ndarray, text: str):
        """
        Draw text caption on the frame.
        
        Args:
            frame: Frame to draw on
            text: Text to display
        """
        if not text:
            return
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.display_config.get('font_scale', 1.0)
        font_thickness = self.display_config.get('font_thickness', 2)
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Position
        position = self.display_config.get('caption_position', 'bottom')
        h, w = frame.shape[:2]
        
        if position == 'bottom':
            x = (w - text_width) // 2
            y = h - 30
        else:  # top
            x = (w - text_width) // 2
            y = 50
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(
            frame,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, text, (x, y),
            font, font_scale, (255, 255, 255), font_thickness
        )
    
    def run(self, video_source=0):
        """
        Run the real-time recognition system.
        
        Args:
            video_source: Video source (0 for webcam, or path to video file)
        """
        # Open video source
        cap = cv2.VideoCapture(video_source)
        
        # Set video properties if using webcam
        if isinstance(video_source, int):
            video_config = self.config.get('video', {})
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config.get('width', 640))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config.get('height', 480))
            cap.set(cv2.CAP_PROP_FPS, video_config.get('fps', 30))
        
        print("Starting sign language recognition...")
        print("Press 'q' to quit")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, prediction = self.process_frame(frame)
                
                # Display
                if self.display_config.get('show_video', True):
                    cv2.imshow('Sign Language Recognition', processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.close()
            if self.tts:
                self.tts.stop()
    
    def reset(self):
        """Reset the recognizer state."""
        self.sequence.clear()
        self.current_text = ""
        self.last_prediction = None
