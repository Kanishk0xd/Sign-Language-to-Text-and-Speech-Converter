# Quick Start Guide

Get up and running with Sign Language Recognition in 5 minutes!

## Prerequisites

- Python 3.8+
- Webcam (for real-time recognition)
- pip

## Installation

```bash
# Clone repository
git clone https://github.com/Kanishk0xd/Sign-Language-to-Text-and-Speech-Converter.git
cd Sign-Language-to-Text-and-Speech-Converter

# Install dependencies
pip install -r requirements.txt
```

## Test the System

```bash
# Run the demo to verify everything works
python example_demo.py

# Run the comprehensive test suite
python test_system.py
```

## Basic Usage

### 1. Real-Time Recognition (Demo Mode)

```bash
# Start with webcam (creates demo model automatically)
python main.py
```

The system will:
- ✓ Open your webcam
- ✓ Detect hand landmarks
- ✓ Show live video with captions
- ✓ Speak detected signs (if TTS available)

Press 'q' to quit.

### 2. Collect Training Data

```bash
# Collect data for a sign (e.g., "hello")
python collect_data.py --sign hello --sequences 30
```

Instructions:
- Press SPACE to start recording
- Perform the sign during the 3-second countdown
- Repeat for the target number of sequences

Repeat for each sign you want to recognize.

### 3. Train Your Model

```bash
# Train with collected data
python train_model.py --data-dir data --output my_model.h5 --epochs 50
```

### 4. Use Your Trained Model

```bash
# Run with your trained model
python main.py --model my_model.h5
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Essential settings
video:
  source: 0          # Webcam (0, 1, 2...) or video file path

model:
  threshold: 0.8     # Confidence threshold (0.0-1.0)

vocabulary:
  hello: "Hello"
  thanks: "Thank you"
  # Add your signs here

tts:
  enabled: true
  mode: "offline"    # or "online"
```

## Common Commands

```bash
# Demo and testing
python example_demo.py              # Run demo
python test_system.py               # Run tests

# Data collection
python collect_data.py --sign hello --sequences 30
python collect_data.py --sign thanks --sequences 30

# Training
python train_model.py --output model.h5 --epochs 50

# Recognition
python main.py                      # Webcam with demo model
python main.py --model model.h5     # Webcam with trained model
python main.py --video video.mp4    # Video file

# Custom config
python main.py --config my_config.yaml
```

## Troubleshooting

**Problem**: Webcam not detected
```bash
# Try different camera index
python main.py
# Edit config.yaml: video.source: 1 or 2
```

**Problem**: TTS not working
```bash
# For offline TTS, install espeak
sudo apt-get install espeak

# Or use online mode in config.yaml
tts:
  mode: "online"
```

**Problem**: Low performance
```bash
# Edit config.yaml
model:
  sequence_length: 15  # Reduce from 30

video:
  width: 320          # Reduce from 640
  height: 240         # Reduce from 480
```

## What's Next?

1. **Collect more data**: More sequences = better accuracy
2. **Train longer**: More epochs = better model
3. **Add vocabulary**: Expand your sign vocabulary
4. **Fine-tune**: Adjust thresholds and parameters

## Project Structure

```
.
├── main.py              # Main application
├── train_model.py       # Training script
├── collect_data.py      # Data collection
├── example_demo.py      # Demo script
├── test_system.py       # Test suite
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── src/                 # Source code
    ├── hand_tracker.py
    ├── sign_recognizer.py
    ├── models/
    └── utils/
```

## Example Workflow

### Day 1: Setup and Test
```bash
pip install -r requirements.txt
python example_demo.py
python test_system.py
python main.py  # Try demo mode
```

### Day 2: Collect Data
```bash
# Collect 30 sequences for each sign
python collect_data.py --sign hello --sequences 30
python collect_data.py --sign thanks --sequences 30
python collect_data.py --sign yes --sequences 30
python collect_data.py --sign no --sequences 30
```

### Day 3: Train Model
```bash
python train_model.py --data-dir data --output my_model.h5 --epochs 100
```

### Day 4: Use Your Model
```bash
python main.py --model my_model.h5
```

## Tips for Best Results

1. **Good Lighting**: Ensure your hands are well-lit
2. **Clear Background**: Avoid cluttered backgrounds
3. **Consistent Position**: Keep hands in frame
4. **Variety**: Collect data from different angles
5. **Repetition**: More training sequences = better accuracy

## Resources

- **Documentation**: See README.md and USAGE.md
- **Config Reference**: See config.yaml comments
- **Issues**: Report bugs on GitHub

## Need Help?

- Check USAGE.md for detailed documentation
- Run test_system.py to diagnose issues
- Review config.yaml settings
- Open an issue on GitHub

---

**Happy Signing! 🤟**
