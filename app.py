import streamlit as st
import pydicom
import numpy as np
import torch
import os
import monai
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel
)
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import uuid

# Detect CUDA/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Secure storage directory
STORAGE_DIR = "secure_storage"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(STORAGE_DIR, "reports"), exist_ok=True)

# Preprocessing pipeline (using MONAI for medical images)
preprocess_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image"], spatial_size=(256, 256, -1)),  # Handle 3D if needed
])

# Load models with caching (define here, call lazily; removed TinyLlama for memory)
@st.cache_resource
def load_models():
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)  # Suppress warning
        gpt2_tok = GPT2Tokenizer.from_pretrained("distilgpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
        return clip_model, clip_proc, blip_model, blip_proc, gpt2_tok, gpt2_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

# Function to process DICOM
def process_dicom(file_path):
    ds = pydicom.dcmread(file_path)
    image = ds.pixel_array
    modality = ds.Modality  # e.g., CT, MR, US, etc.
    return image, modality, ds

# Function to process JPEG
def process_jpeg(file_path):
    image = np.array(Image.open(file_path).convert("L"))  # Convert to grayscale numpy array
    modality = "JPEG"
    metadata = {}  # No metadata for JPEG
    return image, modality, metadata

# Multimodal fusion placeholder (integrates techniques from papers)
def multimodal_fusion(images, metadata, notes):
    # Placeholder: Fuse using transformers/MHSA/cross-attention
    fused_features = np.random.rand(512)  # Mock fused vector
    return fused_features

# Analysis and reasoning placeholder
def analyze_and_reason(fused_features):
    # Placeholder: Multi-round reasoning with LLM
    return "Diagnostic insights: Potential anomaly in region X."

# Report generation adapted from old code (use GPT2 for draft too)
def generate_report(analysis, images, clip_model, clip_proc, blip_model, blip_proc, gpt2_tok, gpt2_model):
    # Reference findings (from old code; expand for multi-modality)
    reference_findings = [
        "There is evidence of right lower lobe consolidation.",
        "No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly with clear lungs.",
        "Findings consistent with pneumonia.",
        "Chronic interstitial changes noted."
    ]
    
    # Generate captions for each image (handle multi-modality)
    captions = []
    for img_tensor in images:
        # Convert MONAI tensor to PIL (take first channel/slice if multi-dim)
        if img_tensor.ndim > 2:
            img_array = img_tensor[0].cpu().numpy()  # First channel
        else:
            img_array = img_tensor.cpu().numpy()
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6) * 255
        img_pil = Image.fromarray(img_array.astype(np.uint8)).convert("RGB").resize((224, 224))
        
        # Caption with BLIP
        blip_inputs = blip_proc(images=img_pil, return_tensors="pt").to(device)
        caption_ids = blip_model.generate(**blip_inputs)
        caption = blip_proc.decode(caption_ids[0], skip_special_tokens=True)
        captions.append(caption)
    
    caption_str = " ".join(captions) + f" Analysis: {analysis}"
    
    # CLIP similarity (use first image for features; or average if multi)
    img_pil = Image.fromarray(img_array.astype(np.uint8)).convert("RGB").resize((224, 224))  # Use last for simplicity
    img_inputs = clip_proc(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        img_features = clip_model.get_image_features(**img_inputs)
        img_features = torch.nn.functional.normalize(img_features, dim=-1)
    
    text_inputs = clip_proc(text=reference_findings, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        txt_features = clip_model.get_text_features(**text_inputs)
        txt_features = torch.nn.functional.normalize(txt_features, dim=-1)
    
    similarities = torch.matmul(txt_features, img_features.T).squeeze()
    top_indices = torch.topk(similarities, k=3).indices
    top_findings = [reference_findings[i] for i in top_indices]
    
    # Draft with GPT2 (replaced TinyLlama)
    prompt = (
        f"🧠 Caption: {caption_str}\n"
        f"📚 Retrieved Reports:\n" +
        "\n".join(f"- {f}" for f in top_findings) +
        "\n\nGenerate a clinical-style radiology report:"
    )
    inputs = gpt2_tok(prompt, return_tensors="pt").to(device)
    output = gpt2_model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    draft_report = gpt2_tok.decode(output[0], skip_special_tokens=True)
    
    # Refine with GPT2 (first pass)
    gpt2_input_1 = f"Caption: {caption_str}\nDraft: {draft_report}\nRefine this into a structured radiology report:"
    gpt2_inputs_1 = gpt2_tok(gpt2_input_1, return_tensors="pt").to(device)
    gpt2_output_1 = gpt2_model.generate(**gpt2_inputs_1, max_new_tokens=250, do_sample=True)
    refined_1 = gpt2_tok.decode(gpt2_output_1[0], skip_special_tokens=True)
    
    # Refine with GPT2 (second pass)
    gpt2_input_2 = f"{refined_1}\nRefine the new report:"
    gpt2_inputs_2 = gpt2_tok(gpt2_input_2, return_tensors="pt").to(device)
    gpt2_output_2 = gpt2_model.generate(**gpt2_inputs_2, max_new_tokens=200, do_sample=True)
    report = gpt2_tok.decode(gpt2_output_2[0], skip_special_tokens=True)
    
    # Mock Grad-CAM visualization (using first image)
    fig, ax = plt.subplots()
    ax.imshow(img_array, cmap="gray")
    ax.set_title("Grad-CAM Heatmap (Mock)")
    plt.savefig("heatmap.png")
    
    return report, "heatmap.png"

# Streamlit App
st.title("AI Doctor Companion: Multi-Modality Medical Image Analysis")

uploaded_files = st.file_uploader("Upload DICOM or JPEG files (Multi-Modality: CT, MRI, X-ray, Ultrasound, PET, Endoscopy)", type=["dcm", "jpg", "jpeg"], accept_multiple_files=True)
clinical_notes = st.text_area("Enter Clinical Notes")

if st.button("Process and Generate Report") and uploaded_files:
    # Lazy load models here
    clip_model, clip_proc, blip_model, blip_proc, gpt2_tok, gpt2_model = load_models()
    
    images = []
    modalities = []
    metadata_list = []
    for uploaded_file in uploaded_files:
        unique_id = str(uuid.uuid4())
        save_path = os.path.join(STORAGE_DIR, "images", f"{unique_id}_{uploaded_file.name}")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == '.dcm':
            _, modality, metadata = process_dicom(save_path)
        elif file_ext in ['.jpg', '.jpeg']:
            _, modality, metadata = process_jpeg(save_path)
        else:
            st.error(f"Unsupported file type: {file_ext}")
            continue
        
        processed_data = preprocess_transforms({"image": save_path})
        images.append(processed_data["image"])
        modalities.append(modality)
        metadata_list.append(metadata)
    
    # Fusion
    fused = multimodal_fusion(images, metadata_list, clinical_notes)
    
    # Analysis
    analysis = analyze_and_reason(fused)
    
    # Report
    report, heatmap_path = generate_report(analysis, images, clip_model, clip_proc, blip_model, blip_proc, gpt2_tok, gpt2_model)
    
    # Save report
    report_path = os.path.join(STORAGE_DIR, "reports", f"report_{unique_id}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    st.write("**Analysis:**", analysis)
    st.write("**Generated Report:**", report)
    st.image(heatmap_path, caption="Explainability Visualization (Grad-CAM Mock)")
    st.download_button("Download Report", report, file_name="medical_report.txt")
