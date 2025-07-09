from ingestion import ingestion
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipModel
from PIL import Image
import torch
import numpy as np
import faiss


image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

image_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")

image_embeddings = []

pdf_path = "data/test8.pdf"
text_chunks, all_images = ingestion(pdf_path)

# print(text_chunks)


print(f"Total text chunks ingested: {len(text_chunks)}")


text_model = SentenceTransformer("intfloat/e5-base-v2")

text_embeddings = text_model.encode(text_chunks)

for idx, img in enumerate(all_images):
    inputs = image_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        vision_output = image_model.vision_model(**inputs).pooler_output
    embedding = torch.nn.functional.normalize(vision_output, p=2, dim=1)
    image_embeddings.append(embedding[0])
    # print(f"BLIP-encoded image {idx+1}/{len(all_images)}")



text_embeddings_np = np.array(text_embeddings).astype("float32")  # (146, 768)
image_embeddings_np = np.stack([img.numpy() for img in image_embeddings]).astype("float32")  # (num_images, dim)

# Text index
text_index = faiss.IndexFlatL2(text_embeddings_np.shape[1])
text_index.add(text_embeddings_np)

# Image index
image_index = faiss.IndexFlatL2(image_embeddings_np.shape[1])
image_index.add(image_embeddings_np)

faiss.write_index(text_index, "text_index.faiss")
faiss.write_index(image_index, "image_index.faiss")


