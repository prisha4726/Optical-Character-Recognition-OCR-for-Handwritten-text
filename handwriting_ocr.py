import cv2
import numpy as np
import pytesseract
import os


# ==========================================
# 1. SET TESSERACT PATH
# ==========================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

print("Tesseract path set successfully.")


# ==========================================
# 2. IMAGE PATH
# ==========================================

image_path = "handwriting.jpg"

image = cv2.imread(image_path)

if image is None:
    print()
    print("ERROR: handwriting.jpg not found!")
    print()
    print("Current folder:")
    print(os.getcwd())
    print()
    print("Make sure handwriting.jpg is inside the Cv project folder.")
    exit()

print("Image loaded successfully!")


# ==========================================
# 3. RESIZE IMAGE
# ==========================================

print("Resizing image...")

scale = 4

image = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

cv2.imwrite("01_resized.jpg", image)


# ==========================================
# 4. CONVERT TO GRAYSCALE
# ==========================================

print("Converting to grayscale...")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite("02_grayscale.jpg", gray)


# ==========================================
# 5. NOISE REDUCTION
# ==========================================

print("Removing noise...")

blur = cv2.GaussianBlur(
    gray,
    (3, 3),
    0
)

cv2.imwrite("03_blur.jpg", blur)


# ==========================================
# 6. THRESHOLDING
# ==========================================

print("Applying thresholding...")

threshold = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)[1]

cv2.imwrite("04_threshold.jpg", threshold)


# ==========================================
# 7. MORPHOLOGICAL PROCESSING
# ==========================================

print("Enhancing handwriting...")

kernel = np.ones(
    (2, 2),
    np.uint8
)

processed = cv2.morphologyEx(
    threshold,
    cv2.MORPH_CLOSE,
    kernel
)

cv2.imwrite("05_processed.jpg", processed)


# ==========================================
# 8. OCR CONFIGURATION
# ==========================================

print("Configuring OCR...")

custom_config = r"--oem 3 --psm 6"


# ==========================================
# 9. PERFORM OCR
# ==========================================

print()
print("Recognizing handwriting...")
print()

text = pytesseract.image_to_string(
    processed,
    config=custom_config
)


# ==========================================
# 10. CLEAN TEXT
# ==========================================

text = text.strip()

lines = []

for line in text.splitlines():

    line = line.strip()

    if line != "":
        lines.append(line)

text = "\n".join(lines)


# ==========================================
# 11. DISPLAY RESULT
# ==========================================

print()
print("======================================")
print("          RECOGNIZED TEXT")
print("======================================")

if text:
    print(text)
else:
    print("No text detected.")

print("======================================")


# ==========================================
# 12. SAVE TEXT
# ==========================================

with open(
    "recognized_text.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(text)

print()
print("Text saved to:")
print("recognized_text.txt")


# ==========================================
# 13. GET OCR DATA
# ==========================================

print()
print("Detecting individual words...")

data = pytesseract.image_to_data(
    processed,
    config=custom_config,
    output_type=pytesseract.Output.DICT
)


# ==========================================
# 14. DRAW BOUNDING BOXES
# ==========================================

output_image = image.copy()

detected_words = 0

for i in range(len(data["text"])):

    word = data["text"][i].strip()

    if word == "":
        continue

    try:
        confidence = float(data["conf"][i])
    except ValueError:
        continue

    if confidence < 20:
        continue

    x = data["left"][i]
    y = data["top"][i]
    w = data["width"][i]
    h = data["height"][i]

    # Draw bounding box
    cv2.rectangle(
        output_image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Display detected word
    cv2.putText(
        output_image,
        word,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    detected_words += 1

    print(
        "Detected:",
        word,
        "| Confidence:",
        round(confidence, 2),
        "%"
    )


# ==========================================
# 15. SAVE FINAL RESULT
# ==========================================

cv2.imwrite(
    "recognized_result.jpg",
    output_image
)


# ==========================================
# 16. FINAL INFORMATION
# ==========================================

print()
print("======================================")
print("          PROCESS COMPLETED")
print("======================================")
print()

print("Detected words:", detected_words)

print()

print("Output files:")
print("1. 01_resized.jpg")
print("2. 02_grayscale.jpg")
print("3. 03_blur.jpg")
print("4. 04_threshold.jpg")
print("5. 05_processed.jpg")
print("6. recognized_text.txt")
print("7. recognized_result.jpg")

print()
print("Computer Vision OCR completed!")