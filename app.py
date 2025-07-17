import streamlit as st
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
import io

# Load CLIP model and processor once
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def extract_frames(video_bytes, interval_sec=1):
    import tempfile
    import cv2

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_file.write(video_bytes)
    temp_file.flush()

    cap = cv2.VideoCapture(temp_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval_sec)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()
    return frames


def save_temp_video_and_extract_frames(video_bytes, interval_sec=1):
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=True)
    temp_file.write(video_bytes)
    temp_file.flush()

    cap = cv2.VideoCapture(temp_file.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    count = 0
    interval_frames = int(fps * interval_sec)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval_frames == 0:
            # Convert BGR to RGB and PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()
    return frames

def get_image_embedding(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu()

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs)
    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings.cpu()

def cosine_similarity(a, b):
    return (a @ b.T).squeeze()

# Streamlit UI
st.title("Video Frame Retrieval with CLIP")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

query = st.text_input("Enter your query text")

if uploaded_video is not None and query:
    st.write("Processing video and query, please wait...")

    video_bytes = uploaded_video.read()
    frames = save_temp_video_and_extract_frames(video_bytes, interval_sec=1)

    if not frames:
        st.write("No frames extracted. Please try a different video.")
    else:
        st.write(f"Extracted {len(frames)} frames.")

        # Get embeddings
        image_embeddings = get_image_embedding(frames)
        text_embedding = get_text_embedding(query)

        # Compute similarity
        sims = cosine_similarity(text_embedding, image_embeddings)

        # Get top-k results (e.g., top 3)
        k = 3
        topk_indices = sims.topk(k).indices.numpy()

        st.write(f"Top {k} matching frames:")

        for idx in topk_indices:
            sim_score = sims[idx].item()
            st.image(frames[idx], caption=f"Similarity: {sim_score:.3f}")

