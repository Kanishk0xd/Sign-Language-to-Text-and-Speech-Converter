"""MediaPipe-based hand tracking module for sign language detection."""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    """Extracts hand landmarks from video frames using MediaPipe."""
    
    def __init__(self, max_num_hands=2, min_detection_confidence=0.7, 
                 min_tracking_confidence=0.5):
        """
        Initialize the hand tracker.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.max_num_hands = max_num_hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            landmarks: Flattened array of normalized (x, y, z) coordinates
            results: MediaPipe results object for drawing
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Extract landmarks
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        # If no hands detected, return zeros
        if not landmarks:
            # 21 landmarks per hand * 3 coordinates * max_num_hands
            landmarks = [0.0] * (21 * 3 * self.max_num_hands)
        
        # Pad if fewer hands than max
        expected_length = 21 * 3 * self.max_num_hands
        while len(landmarks) < expected_length:
            landmarks.extend([0.0] * (21 * 3))
        
        # Truncate if more data than expected
        landmarks = landmarks[:expected_length]
        
        return np.array(landmarks, dtype=np.float32), results
    
    def draw_landmarks(self, frame, results):
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: BGR image from OpenCV
            results: MediaPipe results object
            
        Returns:
            frame: Frame with landmarks drawn
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame
    
    def close(self):
        """Release resources."""
        self.hands.close()
