import os
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
from models.protonet import ProtoNet
from utils.helpers import compute_prototypes, predict_class
from utils.data_loader import load_image_tensor

# Load configuration
MODEL_PATH = "models/protonet_best.pth"
SUPPORT_PATH = "data/demo_support"
N_WAY = 3
K_SHOT = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ProtoNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load support set and compute prototypes
def prepare_support():
    support_images = []
    support_labels = []
    label_map = {}

    label_folders = sorted(os.listdir(SUPPORT_PATH))
    for idx, class_folder in enumerate(label_folders):
        class_dir = os.path.join(SUPPORT_PATH, class_folder)
        for img_file in os.listdir(class_dir)[:K_SHOT]:
            img_path = os.path.join(class_dir, img_file)
            image_tensor = load_image_tensor(img_path, transform=transform).unsqueeze(0)
            support_images.append(image_tensor)
            support_labels.append(torch.tensor(idx))
        label_map[idx] = class_folder

    support_images = torch.cat(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    support_embeddings = model(support_images)
    prototypes = compute_prototypes(support_embeddings, support_labels)

    return prototypes, label_map

prototypes, label_map = prepare_support()

# Inference function
def classify_query(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
        predicted_idx = predict_class(prototypes, embedding)[0].item()
    return f"Predicted Class: {label_map[predicted_idx]}"

# Gradio UI
demo_ui = gr.Interface(
    fn=classify_query,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Few-Shot Skin Lesion Classifier",
    description="Upload a skin lesion image to classify it using few-shot learning (ProtoNet).",
    theme="soft"
)

if __name__ == "__main__":
    demo_ui.launch()
