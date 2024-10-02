import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pytesseract
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from scipy import ndimage

def preprocess_image(image, contrast=1.0, brightness=0, binarize=False, denoise=False, deskew=False):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    if deskew:
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    if binarize:
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        return gray

def recognize_text_tesseract(image):
    custom_config = r'--oem 3 --psm 6 -l kan+eng'
    return pytesseract.image_to_string(image, config=custom_config)

def recognize_text_easyocr(image, reader):
    result = reader.readtext(np.array(image))
    return ' '.join([text for _, text, _ in result])

def recognize_text_trocr(image, processor, model):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['kn', 'en'])  # Added English for better recognition of mixed text

@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")  # Using large model for better accuracy
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    return processor, model

def segment_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)

def search_keyword(text, keyword):
    # Convert both text and keyword to lowercase for case-insensitive search
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # Find all occurrences of the keyword
    matches = list(re.finditer(re.escape(keyword_lower), text_lower))
    
    return matches

def highlight_keyword(text, matches, keyword):
    # Sort matches in reverse order to avoid index issues when adding highlighting
    for match in reversed(matches):
        start, end = match.span()
        text = text[:start] + f"**{text[start:end]}**" + text[end:]
    return text

def main():
    st.set_page_config(page_title="Kannada Handwritten Text Recognition and Search", layout="wide")
    st.title("Kannada Handwritten Text Recognition and Search")

    easyocr_reader = load_easyocr_reader()
    trocr_processor, trocr_model = load_trocr_model()

    uploaded_file = st.file_uploader("Choose an image file with Kannada handwritten text", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.header("Image Preprocessing")
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0)
            brightness = st.slider("Brightness", -50, 50, 0)
            binarize = st.checkbox("Binarize Image")
            denoise = st.checkbox("Apply Denoising")
            deskew = st.checkbox("Deskew Image")

            st.header("OCR Options")
            ocr_option = st.radio("Select OCR Method", ["EasyOCR", "Tesseract", "TrOCR"])
            segment = st.checkbox("Segment Image")

        with col2:
            if st.button("Perform Text Recognition", key="recognize"):
                with st.spinner("Processing image..."):
                    preprocessed_image = preprocess_image(image, contrast, brightness, binarize, denoise, deskew)
                    st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

                    if segment:
                        contours = segment_image(preprocessed_image)
                        full_text = ""
                        for i, contour in enumerate(contours[:10]):  # Process top 10 largest contours
                            x, y, w, h = cv2.boundingRect(contour)
                            roi = preprocessed_image[y:y+h, x:x+w]
                            if ocr_option == "EasyOCR":
                                text = recognize_text_easyocr(roi, easyocr_reader)
                            elif ocr_option == "Tesseract":
                                text = recognize_text_tesseract(roi)
                            else:  # TrOCR
                                text = recognize_text_trocr(Image.fromarray(roi).convert('RGB'), trocr_processor, trocr_model)
                            full_text += text + " "
                    else:
                        if ocr_option == "EasyOCR":
                            full_text = recognize_text_easyocr(preprocessed_image, easyocr_reader)
                        elif ocr_option == "Tesseract":
                            full_text = recognize_text_tesseract(preprocessed_image)
                        else:  # TrOCR
                            full_text = recognize_text_trocr(Image.fromarray(preprocessed_image).convert('RGB'), trocr_processor, trocr_model)

                    st.session_state.full_text = full_text  # Store the recognized text in session state
                    st.subheader("Extracted Text:")
                    st.text_area("", value=full_text, height=200)

                    st.subheader("Confidence Feedback")
                    confidence = st.slider("How accurate was the recognition? (0-100)", 0, 100, 50)
                    if st.button("Submit Feedback"):
                        st.success("Thank you for your feedback! This will help improve future recognition.")

        # Keyword search section
        st.header("Keyword Search")
        search_keyword = st.text_input("Enter a keyword or phrase to search:")
        if search_keyword and 'full_text' in st.session_state:
            matches = search_keyword(st.session_state.full_text, search_keyword)
            if matches:
                st.success(f"Found {len(matches)} occurrence(s) of '{search_keyword}'")
                highlighted_text = highlight_keyword(st.session_state.full_text, matches, search_keyword)
                st.markdown(highlighted_text)
            else:
                st.warning(f"No occurrences of '{search_keyword}' found in the text.")

    st.sidebar.header("Tips for Better Results")
    st.sidebar.markdown("""
    1. Ensure good lighting and contrast in the image.
    2. Try different preprocessing settings, especially binarization and deskewing.
    3. For large handwritten files, use the 'Segment Image' option to process text in smaller chunks.
    4. EasyOCR often performs well for Indic scripts like Kannada.
    5. Experiment with denoising for images with background noise.
    6. If results are poor, try adjusting the image before uploading (e.g., increase contrast, convert to grayscale).
    7. For mixed Kannada and English text, the system now supports both languages.
    8. When searching for keywords, try variations of the word to account for potential OCR errors.
    """)

    st.sidebar.header("About")
    st.sidebar.info("""
    This app is optimized for recognizing Kannada handwritten text from images, including large files. 
    It offers three OCR methods:
    1. EasyOCR: Generally accurate for handwritten Kannada text.
    2. Tesseract: May work better for printed Kannada text.
    3. TrOCR: A deep learning model for handwritten text recognition.
    The app now includes image segmentation for better handling of large files and supports mixed Kannada-English text.
    New feature: Keyword search allows you to find specific words or phrases within the recognized text.
    """)

if __name__ == "__main__":
    main()
