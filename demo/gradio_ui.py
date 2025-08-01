import os
import torch
import gradio as gr
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from models.protonet import ProtoNet
from utils.helpers import compute_prototypes, predict_class
from utils.data_loader import prepare_support_query_tensors

# Configuration
MODEL_PATH = "models/protonet_best.pth"
N_WAY = 3
K_SHOT = 5
QUERY = 1
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ProtoNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Store support images per class
support_images = {f"class_{i}": [] for i in range(N_WAY)}

def add_support_image(class_id, image):
    if len(support_images[f"class_{class_id}"]) >= K_SHOT:
        return f"❌ Class {class_id} already has {K_SHOT} images."
    support_images[f"class_{class_id}"].append(transform(image))
    return f"✅ Image added to Class {class_id}."

def reset_support_images():
    for key in support_images:
        support_images[key] = []
    return "🗑️ All support images reset."

def classify(query_img):
    if any(len(support_images[c]) < K_SHOT for c in support_images):
        return "⚠️ Please upload exactly 5 images for each of the 3 classes."
    
    query_tensor = transform(query_img).unsqueeze(0).to(DEVICE)

    support_tensors = []
    support_labels = []
    for class_id, images in support_images.items():
        tensor_stack = torch.stack(images).to(DEVICE)
        support_tensors.append(tensor_stack)
        support_labels.extend([int(class_id[-1])] * K_SHOT)

    support_tensor = torch.cat(support_tensors)
    support_labels = torch.tensor(support_labels).to(DEVICE)

    with torch.no_grad():
        support_embeddings = model(support_tensor)
        query_embedding = model(query_tensor)
        prototypes = compute_prototypes(support_embeddings, support_labels)
        pred = predict_class(prototypes, query_embedding)[0].item()

    return f"🎯 Predicted Class: Class {pred}"

# Gradio UI
support_uploads = [
    gr.Image(label=f"Class {i}", type="pil") for i in range(N_WAY)
]

support_buttons = [
    gr.Button(f"Add to Class {i}") for i in range(N_WAY)
]

with gr.Blocks(title="Few-Shot Skin Lesion Classifier") as demo:
    gr.Markdown("# 🧠 Few-Shot Skin Lesion Classification (ProtoNet)")
    gr.Markdown("Upload 5 support images per class (3 classes), then test with a query image.")
    
    with gr.Row():
        for i in range(N_WAY):
            with gr.Column():
                support_uploads[i].render()
                support_buttons[i].render()

    with gr.Row():
        reset_btn = gr.Button("🔄 Reset All Support Sets")
        reset_output = gr.Textbox(label="Support Status")
    
    with gr.Row():
        query_image = gr.Image(label="Query Image", type="pil")
        query_btn = gr.Button("🔍 Classify Query Image")
        prediction_output = gr.Textbox(label="Prediction")

    # Button Callbacks
    for i in range(N_WAY):
        support_buttons[i].click(fn=add_support_image, inputs=[gr.State(i), support_uploads[i]], outputs=reset_output)
    
    reset_btn.click(fn=reset_support_images, outputs=reset_output)
    query_btn.click(fn=classify, inputs=query_image, outputs=prediction_output)

# Launch
if __name__ == "__main__":
    demo.launch()
