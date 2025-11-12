"""Simple demo script showing basic usage of the sign language converter."""

import numpy as np
from src.models import LSTMSignModel
from src.hand_tracker import HandTracker
from src.utils import TTSEngine, load_config


def demo_hand_tracking():
    """Demonstrate hand tracking capabilities."""
    print("="*60)
    print("Demo: Hand Tracking")
    print("="*60)
    
    tracker = HandTracker(max_num_hands=2)
    print(f"✓ Hand tracker initialized")
    print(f"  - Max hands: 2")
    print(f"  - Detection confidence: 0.7")
    print(f"  - Tracking confidence: 0.5")
    
    # Simulate a frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks, _ = tracker.extract_landmarks(dummy_frame)
    print(f"✓ Landmark extraction working")
    print(f"  - Feature vector size: {len(landmarks)}")
    print(f"  - Expected: {21 * 3 * 2} (21 landmarks × 3 coords × 2 hands)")
    
    tracker.close()
    print()


def demo_model():
    """Demonstrate model creation and prediction."""
    print("="*60)
    print("Demo: LSTM Model")
    print("="*60)
    
    # Create model
    sequence_length = 30
    num_features = 126  # 21 * 3 * 2
    num_classes = 4
    
    model = LSTMSignModel(sequence_length=sequence_length, num_features=num_features)
    model.set_classes(['hello', 'thanks', 'yes', 'no'])
    model.build(num_classes)
    
    print(f"✓ LSTM model created")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Features per frame: {num_features}")
    print(f"  - Classes: {num_classes}")
    
    # Test prediction
    dummy_sequence = np.random.rand(sequence_length, num_features).astype(np.float32)
    class_idx, confidence = model.predict(dummy_sequence)
    class_name = model.get_class_name(class_idx)
    
    print(f"✓ Prediction working")
    print(f"  - Predicted class: {class_name}")
    print(f"  - Confidence: {confidence:.2%}")
    print()


def demo_tts():
    """Demonstrate text-to-speech capabilities."""
    print("="*60)
    print("Demo: Text-to-Speech")
    print("="*60)
    
    # Offline TTS
    print("Testing offline TTS (pyttsx3)...")
    tts_offline = TTSEngine(mode="offline", rate=150, volume=1.0)
    print(f"✓ Offline TTS initialized")
    print(f"  - Mode: offline (pyttsx3)")
    print(f"  - Rate: 150 wpm")
    
    # Note: We won't actually speak in this demo to avoid noise
    print(f"  - Ready to speak: Yes")
    
    # Online TTS
    print("\nTesting online TTS (gTTS)...")
    tts_online = TTSEngine(mode="online", language="en")
    print(f"✓ Online TTS initialized")
    print(f"  - Mode: online (gTTS)")
    print(f"  - Language: en")
    print()


def demo_config():
    """Demonstrate configuration loading."""
    print("="*60)
    print("Demo: Configuration")
    print("="*60)
    
    try:
        config = load_config('config.yaml')
        print(f"✓ Configuration loaded from config.yaml")
        print(f"\nKey settings:")
        print(f"  - Video source: {config.get('video', {}).get('source', 'N/A')}")
        print(f"  - Model type: {config.get('model', {}).get('type', 'N/A')}")
        print(f"  - Sequence length: {config.get('model', {}).get('sequence_length', 'N/A')}")
        print(f"  - TTS enabled: {config.get('tts', {}).get('enabled', 'N/A')}")
        print(f"  - TTS mode: {config.get('tts', {}).get('mode', 'N/A')}")
        
        vocab = config.get('vocabulary', {})
        print(f"  - Vocabulary size: {len(vocab)} signs")
        print(f"  - Signs: {list(vocab.keys())[:5]}...")
        print()
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  Sign Language to Text and Speech Converter - Demo".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print("\n")
    
    try:
        demo_config()
        demo_hand_tracking()
        demo_model()
        demo_tts()
        
        print("="*60)
        print("All Demos Completed Successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run 'python main.py' to start real-time recognition")
        print("  2. Train a model with 'python train_model.py'")
        print("  3. Customize config.yaml for your needs")
        print()
        
    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
