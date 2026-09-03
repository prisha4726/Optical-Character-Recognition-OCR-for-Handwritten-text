# Handwriting OCR using OpenCV and Tesseract

A Python Computer Vision project that extracts handwritten text from an image using OpenCV preprocessing and Tesseract OCR.

## Features

- Resizes the input image
- Converts image to grayscale
- Removes noise with Gaussian blur
- Applies Otsu thresholding
- Enhances handwriting using morphological operations
- Extracts text with Tesseract OCR
- Draws bounding boxes around detected words
- Saves recognized text and output images

## Requirements

- Python 3.x
- Tesseract OCR installed
- OpenCV
- NumPy
- pytesseract

## Installation

```bash
pip install -r requirements.txt
```

Install Tesseract OCR for Windows, then update this line in the Python file if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Usage

Place your input image in the project folder and name it:

```text
handwriting.jpg
```

Run:

```bash
python handwriting_ocr.py
```

## Output

The program creates:

- `recognized_text.txt`
- `recognized_result.jpg`
- Intermediate processed images for OCR analysis
