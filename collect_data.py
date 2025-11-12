"""Data collection script for training custom sign language models."""

import argparse
import cv2
import numpy as np
import os
import json
from datetime import datetime

from src.hand_tracker import HandTracker
from src.utils import load_config


def collect_sign_data(sign_name, num_sequences, sequence_length, output_dir, config):
    """
    Collect training data for a specific sign.
    
    Args:
        sign_name: Name of the sign to collect
        num_sequences: Number of sequences to collect
        sequence_length: Number of frames per sequence
        output_dir: Directory to save collected data
        config: Configuration dictionary
    """
    # Initialize hand tracker
    hands_config = config.get('hands', {})
    tracker = HandTracker(
        max_num_hands=hands_config.get('max_num_hands', 2),
        min_detection_confidence=hands_config.get('min_detection_confidence', 0.7),
        min_tracking_confidence=hands_config.get('min_tracking_confidence', 0.5)
    )
    
    # Initialize video capture
    video_config = config.get('video', {})
    cap = cv2.VideoCapture(video_config.get('source', 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config.get('width', 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config.get('height', 480))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    sign_dir = os.path.join(output_dir, sign_name)
    os.makedirs(sign_dir, exist_ok=True)
    
    collected_sequences = []
    sequence_count = 0
    
    print(f"\n{'='*60}")
    print(f"Collecting data for sign: {sign_name}")
    print(f"{'='*60}")
    print(f"Target: {num_sequences} sequences of {sequence_length} frames each")
    print(f"Output: {sign_dir}")
    print(f"\nInstructions:")
    print("  - Press SPACE to start recording a sequence")
    print("  - Perform the sign while recording")
    print("  - Press 'q' to quit")
    print(f"{'='*60}\n")
    
    recording = False
    current_sequence = []
    countdown = 0
    
    try:
        while sequence_count < num_sequences:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            landmarks, results = tracker.extract_landmarks(frame)
            
            # Draw landmarks
            frame = tracker.draw_landmarks(frame, results)
            
            # Recording logic
            if recording:
                current_sequence.append(landmarks)
                
                # Show recording progress
                progress = len(current_sequence)
                cv2.putText(
                    frame, 
                    f"RECORDING: {progress}/{sequence_length}", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 0, 255), 
                    2
                )
                
                # Draw progress bar
                bar_width = 400
                bar_height = 20
                bar_x = (frame.shape[1] - bar_width) // 2
                bar_y = 50
                
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (255, 255, 255),
                    2
                )
                
                progress_width = int((progress / sequence_length) * bar_width)
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + progress_width, bar_y + bar_height),
                    (0, 255, 0),
                    -1
                )
                
                # Check if sequence complete
                if len(current_sequence) >= sequence_length:
                    collected_sequences.append(np.array(current_sequence))
                    sequence_count += 1
                    recording = False
                    current_sequence = []
                    print(f"✓ Sequence {sequence_count}/{num_sequences} collected")
                    
            else:
                if countdown > 0:
                    # Show countdown
                    cv2.putText(
                        frame, 
                        f"Starting in {countdown}...", 
                        (frame.shape[1]//2 - 100, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, 
                        (0, 255, 255), 
                        3
                    )
                    countdown -= 1
                    if countdown == 0:
                        recording = True
                else:
                    # Show status
                    cv2.putText(
                        frame, 
                        f"Sign: {sign_name} | Collected: {sequence_count}/{num_sequences}", 
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    cv2.putText(
                        frame, 
                        "Press SPACE to start recording", 
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        1
                    )
            
            # Display frame
            cv2.imshow('Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and not recording and countdown == 0:
                countdown = 3  # 3-second countdown
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
    
    # Save collected data
    if collected_sequences:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as numpy array
        data_array = np.array(collected_sequences)
        npy_path = os.path.join(sign_dir, f"{sign_name}_{timestamp}.npy")
        np.save(npy_path, data_array)
        
        # Save metadata
        metadata = {
            'sign_name': sign_name,
            'num_sequences': len(collected_sequences),
            'sequence_length': sequence_length,
            'num_features': data_array.shape[2],
            'timestamp': timestamp,
            'shape': data_array.shape
        }
        
        meta_path = os.path.join(sign_dir, f"{sign_name}_{timestamp}.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Data saved:")
        print(f"  - Array: {npy_path}")
        print(f"  - Metadata: {meta_path}")
        print(f"  - Shape: {data_array.shape}")
    else:
        print("\n✗ No data collected")
    
    return len(collected_sequences)


def main():
    """Main data collection entry point."""
    parser = argparse.ArgumentParser(
        description='Collect training data for sign language recognition'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--sign',
        type=str,
        required=True,
        help='Name of the sign to collect (e.g., hello, thanks)'
    )
    parser.add_argument(
        '--sequences',
        type=int,
        default=30,
        help='Number of sequences to collect (default: 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for collected data (default: data)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Warning: Configuration file '{args.config}' not found.")
        print("Using default configuration.")
        config = {
            'video': {'source': 0, 'width': 640, 'height': 480},
            'hands': {'max_num_hands': 2, 'min_detection_confidence': 0.7},
            'model': {'sequence_length': 30}
        }
    
    sequence_length = config.get('model', {}).get('sequence_length', 30)
    
    # Collect data
    collected = collect_sign_data(
        sign_name=args.sign,
        num_sequences=args.sequences,
        sequence_length=sequence_length,
        output_dir=args.output,
        config=config
    )
    
    if collected > 0:
        print(f"\n{'='*60}")
        print(f"SUCCESS! Collected {collected} sequences for '{args.sign}'")
        print(f"{'='*60}")
        print("\nNext steps:")
        print(f"  1. Collect data for other signs")
        print(f"  2. Run: python train_model.py --data-dir {args.output}")
    else:
        print("\nData collection incomplete. Please try again.")


if __name__ == '__main__':
    main()
