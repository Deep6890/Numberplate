import streamlit as st
import cv2
import numpy as np
import pytesseract
import os

# Configure pytesseract for cloud deployment
if os.path.exists("/usr/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
elif os.path.exists("C:\\Program Files\\Tesseract-OCR\\tesseract.exe"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image(image):
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(grayImg, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    guassBlure = cv2.GaussianBlur(dilated, (5,5), 0)
    edges = cv2.Canny(guassBlure, 280, 400)
    bilateral = cv2.bilateralFilter(edges, 13, 55, 55)
    
    _, thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = thresh.astype("uint8")
    
    ys, xs = np.where(thresh == 255)
    
    if len(xs) == 0 or len(ys) == 0:
        return None, None, "❌ No plate detected"
    
    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)
    
    cropped = grayImg[y1:y2+1, x1:x2+1]
    
    if cropped is None or cropped.size == 0:
        return None, None, "❌ No plate detected"
    
    edges_crop = cv2.Canny(cropped, 60, 200)
    cnts, _ = cv2.findContours(edges_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    polygon_ok = False
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) == 4:
            polygon_ok = True
            break
    
    h, w = cropped.shape
    aspect_ratio = w / float(h)
    edge_density = np.sum(edges_crop > 0) / (h * w)
    mean_brightness = np.mean(cropped)
    
    score = 0
    if polygon_ok: score += 1
    if 2.5 < aspect_ratio < 6.5: score += 1
    if 0.02 < edge_density < 0.25: score += 1
    if mean_brightness > 90: score += 1
    
    result = "🟩 NUMBER PLATE" if score >= 3 else "❌ NOT A PLATE"
    
    return cropped, (x1, y1, x2, y2), result

st.title("Vehicle Number Plate Detector")
st.write("Upload an image to detect the number plate and read the text.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png","webp"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    draw_img = image.copy()
    
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             caption="Uploaded Image",
             use_column_width=True)
    
    cropped, coords, result = process_image(image)
    
    st.header("Detection Result")
    st.write(result)
    
    if cropped is not None and coords is not None:
        x1, y1, x2, y2 = coords
        
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0,255,0), 3)
        
        st.image(
            cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB),
            caption="Detected Plate on Original Image",
            use_column_width=True
        )
        
        gray = cropped
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41, 2
        )
        
        st.image(thresh, caption="Preprocessed for OCR", use_column_width=True)
        
        config = "-l eng --oem 3 --psm 7"
        
        try:
            raw_text = pytesseract.image_to_string(thresh, config=config)
            text = "".join(ch for ch in raw_text.upper() if ch.isalnum())
            
            st.subheader("Extracted Text:")
            if text:
                st.code(text)
            else:
                st.warning("OCR failed — try clearer image.")
        except Exception as e:
            st.error("OCR not available in this environment")