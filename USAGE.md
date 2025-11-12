# Usage Guide

This guide explains how to use the Sign Language to Text and Speech Converter.

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Kanishk0xd/Sign-Language-to-Text-and-Speech-Converter.git
cd Sign-Language-to-Text-and-Speech-Converter

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
# Test that everything is working
python example_demo.py
```

### 3. Start Real-Time Recognition

```bash
# Using webcam (default)
python main.py

# Using a video file
python main.py --video path/to/video.mp4
```

## Features Overview

### Real-Time Recognition

The system processes video frames in real-time:
1. Detects hands using MediaPipe
2. Extracts 21 landmark points per hand
3. Collects sequences of frames
4. Predicts signs using LSTM model
5. Displays text captions
6. Optionally speaks the text

### Privacy-First Design

All processing happens on your device:
- No video is uploaded to servers
- No internet required (with offline TTS)
- MediaPipe runs locally
- Model inference is on-device

### Configurable System

Edit `config.yaml` to customize:

```yaml
# Video input
video:
  source: 0  # 0=webcam, or path to video file
  width: 640
  height: 480

# Model behavior
model:
  sequence_length: 30  # Frames per prediction
  threshold: 0.8       # Confidence threshold

# Text-to-Speech
tts:
  enabled: true
  mode: "offline"      # or "online"
  rate: 150            # Words per minute

# Vocabulary
vocabulary:
  hello: "Hello"
  thanks: "Thank you"
```

## Training Your Own Model

### Step 1: Collect Data

Create a data collection script or use the system in "recording mode":

```python
from src.hand_tracker import HandTracker
import cv2
import numpy as np

# Initialize tracker
tracker = HandTracker()
cap = cv2.VideoCapture(0)

sequences = []
for i in range(100):  # Collect 100 sequences
    sequence = []
    for j in range(30):  # 30 frames per sequence
        ret, frame = cap.read()
        landmarks, _ = tracker.extract_landmarks(frame)
        sequence.append(landmarks)
    sequences.append(sequence)

# Save sequences
np.save('sign_data.npy', sequences)
```

### Step 2: Train the Model

```bash
python train_model.py \
    --data-dir path/to/data \
    --output my_model.h5 \
    --epochs 50
```

### Step 3: Use Your Model

```bash
python main.py --model my_model.h5
```

## Advanced Configuration

### Custom Model Architecture

Create your own model by extending `BaseSignModel`:

```python
from src.models import BaseSignModel
import tensorflow as tf

class TransformerSignModel(BaseSignModel):
    def build(self, num_classes):
        # Your architecture here
        model = tf.keras.Sequential([
            # Add Transformer layers
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        # Your training logic
        return self.model.fit(X_train, y_train, epochs=epochs)
```

### Custom Phrase Mapping

Map detected signs to full phrases:

```yaml
vocabulary:
  # Simple mappings
  hello: "Hello"
  thanks: "Thank you"
  
  # Complex phrases
  help_me: "I need assistance"
  bathroom: "Where is the bathroom?"
  hungry: "I am hungry"
```

### Video Processing Settings

Adjust for your hardware:

```yaml
video:
  width: 1280    # Higher resolution
  height: 720
  fps: 60        # Faster capture

hands:
  max_num_hands: 2
  min_detection_confidence: 0.8  # More strict
```

## Troubleshooting

### Low Frame Rate

- Reduce video resolution
- Use a faster model (fewer LSTM layers)
- Ensure GPU is available for TensorFlow

### Poor Recognition

- Improve lighting conditions
- Keep hands clearly visible
- Increase `min_detection_confidence`
- Train with more diverse data
- Lower `threshold` for more predictions

### TTS Issues

**Offline TTS not working:**
- Install espeak: `sudo apt-get install espeak`
- Or switch to online mode

**Online TTS not working:**
- Check internet connection
- Install pygame: `pip install pygame`
- Install gTTS: `pip install gTTS`

## Performance Tips

### CPU Optimization

- Use smaller model architectures
- Reduce `sequence_length`
- Lower video resolution
- Skip frames if needed

### GPU Acceleration

TensorFlow will automatically use GPU if available:

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Memory Management

For long-running sessions:

```python
# In your code
import gc
import tensorflow as tf

# Periodically clear memory
gc.collect()
tf.keras.backend.clear_session()
```

## Examples

### Example 1: Silent Mode

Disable TTS for quiet environments:

```yaml
tts:
  enabled: false
```

### Example 2: High-Accuracy Mode

Stricter detection for fewer false positives:

```yaml
model:
  threshold: 0.95

hands:
  min_detection_confidence: 0.9
```

### Example 3: Demo Mode

Quick setup for demonstrations:

```yaml
model:
  sequence_length: 15  # Faster predictions
  threshold: 0.7       # More responsive

display:
  show_landmarks: true
  caption_position: "bottom"
```

## API Reference

### HandTracker

```python
from src.hand_tracker import HandTracker

tracker = HandTracker(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Extract landmarks from frame
landmarks, results = tracker.extract_landmarks(frame)

# Draw landmarks on frame
frame = tracker.draw_landmarks(frame, results)
```

### SignRecognizer

```python
from src.sign_recognizer import SignRecognizer
from src.models import LSTMSignModel

model = LSTMSignModel()
model.load('my_model.h5')

recognizer = SignRecognizer(model, config)

# Process single frame
processed_frame, prediction = recognizer.process_frame(frame)

# Run continuous recognition
recognizer.run(video_source=0)
```

### TTSEngine

```python
from src.utils import TTSEngine

# Offline TTS
tts = TTSEngine(mode="offline", rate=150, volume=1.0)
tts.speak("Hello world")

# Online TTS
tts = TTSEngine(mode="online", language="en")
tts.speak("Hello world")
```

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Submit a pull request

Areas for contribution:
- Additional model architectures (Transformer, CNN-LSTM)
- Data collection tools
- Pre-trained models for common sign languages
- Support for more TTS engines
- Mobile app integration
- Web interface

## License

MIT License - feel free to use and modify for your needs.
