"""Comprehensive test suite for the sign language recognition system."""

import sys
import os
import numpy as np
import tempfile


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...", end=" ")
    try:
        from src.hand_tracker import HandTracker
        from src.sign_recognizer import SignRecognizer
        from src.models import BaseSignModel, LSTMSignModel
        from src.utils import TTSEngine, load_config
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...", end=" ")
    try:
        from src.utils import load_config
        config = load_config('config.yaml')
        
        # Check required sections
        assert 'video' in config, "Missing 'video' section"
        assert 'model' in config, "Missing 'model' section"
        assert 'hands' in config, "Missing 'hands' section"
        assert 'vocabulary' in config, "Missing 'vocabulary' section"
        assert 'tts' in config, "Missing 'tts' section"
        
        # Check vocabulary
        vocab = config['vocabulary']
        assert isinstance(vocab, dict), "Vocabulary must be a dictionary"
        assert len(vocab) > 0, "Vocabulary is empty"
        
        # Check all vocabulary keys are strings
        for key in vocab.keys():
            assert isinstance(key, str), f"Vocabulary key must be string: {key}"
        
        print(f"✓ ({len(vocab)} vocabulary entries)")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_hand_tracker():
    """Test hand tracking functionality."""
    print("Testing hand tracker...", end=" ")
    try:
        from src.hand_tracker import HandTracker
        
        tracker = HandTracker(max_num_hands=2)
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Extract landmarks
        landmarks, results = tracker.extract_landmarks(frame)
        
        # Verify output
        assert landmarks is not None, "Landmarks is None"
        assert len(landmarks) == 126, f"Expected 126 features, got {len(landmarks)}"
        assert isinstance(landmarks, np.ndarray), "Landmarks must be numpy array"
        
        tracker.close()
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_lstm_model():
    """Test LSTM model functionality."""
    print("Testing LSTM model...", end=" ")
    try:
        from src.models import LSTMSignModel
        
        # Create model
        model = LSTMSignModel(sequence_length=30, num_features=126)
        classes = ['hello', 'thanks', 'yes', 'no']
        model.set_classes(classes)
        model.build(len(classes))
        
        # Test prediction
        sequence = np.random.rand(30, 126).astype(np.float32)
        class_idx, confidence = model.predict(sequence)
        
        assert 0 <= class_idx < len(classes), "Invalid class index"
        assert 0 <= confidence <= 1, "Invalid confidence value"
        
        # Test class name retrieval
        class_name = model.get_class_name(class_idx)
        assert class_name in classes, f"Invalid class name: {class_name}"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_model_save_load():
    """Test model saving and loading."""
    print("Testing model save/load...", end=" ")
    try:
        from src.models import LSTMSignModel
        
        # Create and save model
        model = LSTMSignModel(sequence_length=30, num_features=126)
        classes = ['hello', 'thanks', 'yes', 'no']
        model.set_classes(classes)
        model.build(len(classes))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        model.save(temp_path)
        
        # Load model
        model2 = LSTMSignModel(sequence_length=30, num_features=126)
        model2.load(temp_path)
        model2.set_classes(classes)
        
        # Test prediction with loaded model
        sequence = np.random.rand(30, 126).astype(np.float32)
        class_idx, confidence = model2.predict(sequence)
        
        assert 0 <= class_idx < len(classes), "Invalid class index after load"
        
        # Cleanup
        os.unlink(temp_path)
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_tts_engine():
    """Test TTS engine initialization."""
    print("Testing TTS engine...", end=" ")
    try:
        from src.utils import TTSEngine
        
        # Test offline mode
        tts_offline = TTSEngine(mode='offline', rate=150, volume=1.0)
        assert tts_offline.mode == 'offline', "Offline mode not set"
        
        # Test online mode
        tts_online = TTSEngine(mode='online', language='en')
        assert tts_online.mode == 'online', "Online mode not set"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_sign_recognizer():
    """Test sign recognizer integration."""
    print("Testing sign recognizer...", end=" ")
    try:
        from src.sign_recognizer import SignRecognizer
        from src.models import LSTMSignModel
        from src.utils import load_config
        
        # Load config
        config = load_config('config.yaml')
        
        # Create model
        model = LSTMSignModel(sequence_length=30, num_features=126)
        classes = list(config['vocabulary'].keys())
        model.set_classes(classes)
        model.build(len(classes))
        
        # Create recognizer
        recognizer = SignRecognizer(model, config)
        
        # Test frame processing
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_frame, prediction = recognizer.process_frame(frame)
        
        assert processed_frame.shape == frame.shape, "Frame shape changed"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def test_sequence_buffer():
    """Test sequence buffering in recognizer."""
    print("Testing sequence buffer...", end=" ")
    try:
        from src.sign_recognizer import SignRecognizer
        from src.models import LSTMSignModel
        from src.utils import load_config
        
        config = load_config('config.yaml')
        sequence_length = config.get('model', {}).get('sequence_length', 30)
        
        model = LSTMSignModel(sequence_length=sequence_length, num_features=126)
        classes = list(config['vocabulary'].keys())
        model.set_classes(classes)
        model.build(len(classes))
        
        recognizer = SignRecognizer(model, config)
        
        # Process multiple frames to fill buffer
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(sequence_length + 5):
            recognizer.process_frame(frame)
        
        # Check buffer size
        assert len(recognizer.sequence) == sequence_length, "Buffer size incorrect"
        
        print("✓")
        return True
    except Exception as e:
        print(f"✗ {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SIGN LANGUAGE RECOGNITION SYSTEM - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_configuration,
        test_hand_tracker,
        test_lstm_model,
        test_model_save_load,
        test_tts_engine,
        test_sign_recognizer,
        test_sequence_buffer,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  - Run 'python main.py' to start recognition")
        print("  - Run 'python collect_data.py --sign hello' to collect data")
        print("  - Run 'python train_model.py' to train a model")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
