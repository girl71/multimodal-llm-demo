# -*- coding: utf-8 -*-

!pip install wfdb transformers torchvision torchaudio timm pillow numpy scipy

import wfdb

wfdb.dl_database("mitdb", "./ecg_data")

record = wfdb.rdrecord("./ecg_data/100")
ecg_signal = record.p_signal[:,0]

print("ECG length:", len(ecg_signal))

import numpy as np
from scipy.signal import find_peaks

peaks, _ = find_peaks(ecg_signal, distance=150)


fs = record.fs  # sampling frequency
duration_minutes = len(ecg_signal) / fs / 60
heart_rate = len(peaks) / duration_minutes

sensor_features = {
    "heart_rate": int(heart_rate),
    "ecg_mean": float(np.mean(ecg_signal)),
    "ecg_std": float(np.std(ecg_signal))
}

sensor_features

import torch
from PIL import Image
import timm
import torchvision.transforms as transforms

# Vision Transformer Load
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.eval()

# Prepare Image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("chest.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # [1,3,224,224]

with torch.no_grad():
    image_embedding = vit_model.forward_features(img_tensor)  # [1,768]

print("image embedding shape:", image_embedding.shape)

from transformers import AutoTokenizer, AutoModel

text_model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name)

symptoms = "Patient reports chest pain and dizziness"

inputs = tokenizer(symptoms, return_tensors="pt")

with torch.no_grad():
    text_embedding = text_model(**inputs).last_hidden_state.mean(dim=1)

print("text embedding shape:", text_embedding.shape)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Convert feature ECG to embedding
sensor_tensor = torch.tensor([
    sensor_features["heart_rate"],
    sensor_features["ecg_mean"],
    sensor_features["ecg_std"]
]).float().unsqueeze(0)  # [1,3]

# projection layers
sensor_proj = nn.Linear(3,128)
image_proj = nn.Linear(image_embedding.shape[-1],128)
text_proj = nn.Linear(text_embedding.shape[-1],128)

# Apply projection
sensor_vec = sensor_proj(sensor_tensor)   # [1,128]

# For image_embedding: if is 3D  (1, seq, hidden) → mean over seq
if len(image_embedding.shape) == 3:
    image_vec = image_proj(image_embedding.mean(dim=1))  # [1,128]
else:
    image_vec = image_proj(image_embedding)  # [1,128]

# For text_embedding: if is 3D (1, seq, hidden) → mean over seq
if len(text_embedding.shape) == 3:
    text_vec = text_proj(text_embedding.mean(dim=1))  # [1,128]
else:
    text_vec = text_proj(text_embedding)  # [1,128]

# concat fusion
fusion = torch.cat([sensor_vec, image_vec, text_vec], dim=1)  # [1,384]
print("fusion vector shape:", fusion.shape)

# Convert fusion to prompt
fusion_prompt = f"""
Patient data summary:

Heart rate: {sensor_features["heart_rate"]}
ECG mean: {sensor_features["ecg_mean"]}
Symptoms: {symptoms}

Use this information along with the X-ray embedding to provide possible medical interpretation.
"""

# Load light LLM for light reasoning
llm_name = "facebook/opt-350m"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name)

# encode prompt
inputs = llm_tokenizer(fusion_prompt, return_tensors="pt")

# generate
output = llm.generate(**inputs, max_new_tokens=150)

# decode & show output
print(llm_tokenizer.decode(output[0], skip_special_tokens=True))
