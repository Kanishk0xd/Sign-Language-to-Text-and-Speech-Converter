"""Main entry point for sign language to text and speech converter."""

import argparse
import os
import sys

from src.models import LSTMSignModel
from src.sign_recognizer import SignRecognizer
from src.utils import load_config


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Sign Language to Text and Speech Converter'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path to video file (if not provided, uses webcam)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model file (if not provided, creates demo model)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    
    # Initialize model
    model_config = config.get('model', {})
    sequence_length = model_config.get('sequence_length', 30)
    
    # For demonstration, we'll create a simple model
    # In production, you would load a pre-trained model
    model = LSTMSignModel(sequence_length=sequence_length)
    
    if args.model and os.path.exists(args.model):
        print(f"Loading model from {args.model}")
        model.load(args.model)
        # Load classes from config vocabulary
        model.set_classes(list(config.get('vocabulary', {}).keys()))
    else:
        print("Warning: No trained model provided.")
        print("Creating demo model for testing purposes.")
        print("For production use, train a model and provide it via --model")
        
        # Create a demo model
        classes = list(config.get('vocabulary', {}).keys())
        if not classes:
            classes = ['hello', 'thanks', 'yes', 'no']
        
        model.set_classes(classes)
        model.build(len(classes))
        print(f"Demo model created with {len(classes)} classes: {classes}")
    
    # Determine video source
    video_source = args.video if args.video else config.get('video', {}).get('source', 0)
    if video_source != 0 and not os.path.exists(video_source):
        print(f"Error: Video file '{video_source}' not found.")
        sys.exit(1)
    
    # Initialize recognizer
    recognizer = SignRecognizer(model, config)
    
    # Run recognition
    print("\n" + "="*60)
    print("Sign Language to Text and Speech Converter")
    print("="*60)
    print(f"Video source: {'Webcam' if video_source == 0 else video_source}")
    print(f"TTS enabled: {config.get('tts', {}).get('enabled', True)}")
    print(f"TTS mode: {config.get('tts', {}).get('mode', 'offline')}")
    print(f"Vocabulary: {list(config.get('vocabulary', {}).keys())}")
    print("="*60)
    print("\nPress 'q' to quit\n")
    
    try:
        recognizer.run(video_source)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nThank you for using Sign Language Converter!")


if __name__ == '__main__':
    main()
