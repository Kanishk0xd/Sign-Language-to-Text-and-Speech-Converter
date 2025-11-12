# Sign Language to Text and Speech Converter

Real-time sign language to text and speech in Python using MediaPipe keypoints and a sequence model. On-device, privacy-first, and accessibility-focused.

## Features

✅ **Real-time sign recognition** from webcam or video file  
✅ **Live text captions** with optional synthesized speech (TTS)  
✅ **On-device inference** for privacy (no video leaves your machine)  
✅ **Configurable vocabulary** and custom phrase mapping  
✅ **Offline TTS** (pyttsx3) or **online TTS** (gTTS)  
✅ **Extensible model interface** (LSTM/Transformer or your own)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Webcam/   │────▶│   MediaPipe  │────▶│   Sequence  │
│   Video     │     │ Hand Tracker │     │   Buffer    │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Speech    │◀────│   Phrase     │◀────│    LSTM     │
│   Output    │     │   Mapping    │     │    Model    │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time recognition)
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kanishk0xd/Sign-Language-to-Text-and-Speech-Converter.git
   cd Sign-Language-to-Text-and-Speech-Converter
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### Basic Usage

Run with webcam (default):
```bash
python main.py
```

Run with a video file:
```bash
python main.py --video path/to/video.mp4
```

Use a custom configuration:
```bash
python main.py --config my_config.yaml
```

Load a trained model:
```bash
python main.py --model trained_model.h5
```

### Training a Model

Train a model with dummy data (for testing):
```bash
python train_model.py --output my_model.h5 --epochs 50
```

Train with your own data:
```bash
python train_model.py --data-dir path/to/data --output my_model.h5
```

## Configuration

Edit `config.yaml` to customize the system:

```yaml
# Video settings
video:
  source: 0  # 0 for webcam, or path to video file
  width: 640
  height: 480

# Model settings
model:
  type: "lstm"
  sequence_length: 30
  threshold: 0.8

# Hand tracking
hands:
  max_num_hands: 2
  min_detection_confidence: 0.7

# Vocabulary mapping
vocabulary:
  hello: "Hello"
  thanks: "Thank you"
  yes: "Yes"
  no: "No"

# Text-to-Speech
tts:
  enabled: true
  mode: "offline"  # offline (pyttsx3) or online (gTTS)
  rate: 150
  language: "en"
```

## Project Structure

```
Sign-Language-to-Text-and-Speech-Converter/
├── src/
│   ├── __init__.py
│   ├── hand_tracker.py        # MediaPipe hand tracking
│   ├── sign_recognizer.py     # Main recognition system
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py      # Abstract model interface
│   │   └── lstm_model.py      # LSTM implementation
│   └── utils/
│       ├── __init__.py
│       ├── tts_engine.py      # Text-to-speech engine
│       └── config_loader.py   # Configuration loader
├── main.py                    # Application entry point
├── train_model.py            # Model training script
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
└── README.md
```

## How It Works

1. **Hand Tracking**: MediaPipe detects and tracks hand landmarks (21 points per hand) in real-time
2. **Feature Extraction**: Normalized (x, y, z) coordinates create a feature vector
3. **Sequence Collection**: Frames are buffered into sequences for temporal analysis
4. **Model Prediction**: LSTM model predicts the sign from the sequence
5. **Phrase Mapping**: Detected signs are mapped to configured phrases
6. **Text & Speech Output**: Captions are displayed and optionally spoken

## Extending the System

### Adding Your Own Model

Create a new model by extending `BaseSignModel`:

```python
from src.models import BaseSignModel

class MyCustomModel(BaseSignModel):
    def build(self, num_classes):
        # Implement your architecture
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        # Implement training logic
        pass
```

### Adding New Signs

1. Collect training data for your signs
2. Add sign names to `vocabulary` in `config.yaml`
3. Train a new model with your data
4. Load the model with `--model`

## Privacy & Security

- **On-device processing**: All video processing happens locally
- **No data transmission**: Video never leaves your machine
- **Optional offline mode**: Use offline TTS to avoid internet dependency

## Requirements

- opencv-python >= 4.8.0
- mediapipe >= 0.10.0
- numpy >= 1.24.0
- tensorflow >= 2.13.0
- pyttsx3 >= 2.90
- gTTS >= 2.3.0
- PyYAML >= 6.0

## Troubleshooting

**Issue**: Webcam not detected
- Check if camera is connected and not in use by another application
- Try changing `video.source` to 1 or 2 in config

**Issue**: TTS not working
- For offline mode: Install pyttsx3 and required audio drivers
- For online mode: Install gTTS and pygame, ensure internet connection

**Issue**: Poor recognition accuracy
- Ensure good lighting conditions
- Keep hands clearly visible to camera
- Train model with more diverse data
- Adjust `model.threshold` in config

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe for hand tracking
- TensorFlow for deep learning
- OpenCV for video processing
