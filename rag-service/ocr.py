import io

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCR:
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, image_bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")

            text = pytesseract.image_to_string(image, lang='deu+eng')
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return None

_ocr_service = None

def ocr_service()  -> OCR:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCR()
    return _ocr_service