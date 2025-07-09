import io
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter

def ingestion(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = ""
    image_list = []

    for page in doc:
        # Extract text
        all_text += page.get_text()

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_list.append(image)
            try:
                ocr_text = pytesseract.image_to_string(image)
                all_text += ocr_text
            except Exception as e:
                print(f"OCR failed on image {img_index}: {e}")

    # Extract tables as text using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    all_text += " | ".join(str(cell) for cell in row if cell) + "\n"
    
    
    all_text = all_text.encode("utf-8", errors="ignore").decode("utf-8")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    return text_chunks, image_list



