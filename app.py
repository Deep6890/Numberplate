import streamlit as st
import cv2
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_plate(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(gray, kernel, 1)
    dilated = cv2.dilate(eroded, kernel, 1)
    blur = cv2.GaussianBlur(dilated, (5, 5), 0)
    edges = cv2.Canny(blur, 250, 400)

    bilateral = cv2.bilateralFilter(edges, 15, 55, 55)
    _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.028 * peri, True)

        if len(approx) <= 4:
            x, y, w, h = cv2.boundingRect(approx)
            return image[y:y + h, x:x + w], (x, y, x + w, y + h)

    return None, None


def read_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 2
    )

    config = "-l eng --oem 3 --psm 7"
    raw = pytesseract.image_to_string(thresh, config=config)
    text = "".join(ch for ch in raw.upper() if ch.isalnum())

    return text, thresh


def fallback_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_intensity = np.mean(gray)

    if avg_intensity < 60:
        return ""

    config = "-l eng --oem 3 --psm 6"
    raw = pytesseract.image_to_string(gray, config=config)
    text = "".join(ch for ch in raw.upper() if ch.isalnum())

    return text


st.title("Vehicle Number Plate Detector")
st.write("Upload an image to detect the number plate text.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp","avif"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             caption="Uploaded Image",
             use_container_width=True)

    # Detect plate
    plate, coords = detect_plate(image)

    final_text = ""
    display_img = image.copy()

    if plate is not None:
        x1, y1, x2, y2 = coords
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        text, ocr_view = read_text(plate)

        if len(text) >= 6:
            final_text = text
        else:
            fallback_text = fallback_ocr(image)
            if len(fallback_text) >= 6:
                final_text = fallback_text

    else:
        fallback_text = fallback_ocr(image)
        if len(fallback_text) >= 6:
            final_text = fallback_text
    
    if final_text:
        st.success("Plate Text Detected !!!!!")
        st.code(final_text)

        cv2.putText(display_img, final_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB),
                 caption="OCR Output",
                 use_container_width=True)

    else:
        st.error("No plate detected in the gieven image")
