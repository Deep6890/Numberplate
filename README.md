# Numberplate Recognition System

A Python-based automatic number plate recognition (ANPR) system.

## Features

- Number plate detection and recognition
- Image processing capabilities
- Easy-to-use interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from numberplate import NumberPlateRecognizer

recognizer = NumberPlateRecognizer()
result = recognizer.recognize('path/to/image.jpg')
print(result)
```

## Requirements

- Python 3.7+
- OpenCV
- Tesseract OCR

## License

MIT License