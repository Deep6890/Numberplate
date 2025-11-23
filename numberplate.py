import cv2
import pytesseract
import numpy as np

class NumberPlateRecognizer:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    
    def recognize(self, image_path):
        """Recognize number plate from image"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        plates = self.cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in plates:
            plate_img = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(plate_img, config='--psm 8')
            return text.strip()
        
        return None

if __name__ == "__main__":
    recognizer = NumberPlateRecognizer()
    # Example usage
    # result = recognizer.recognize('sample_image.jpg')
    # print(f"Detected plate: {result}")