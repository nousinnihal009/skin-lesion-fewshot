# 🔬 Few-Shot Skin Lesion Classification Demo

This folder contains demonstration scripts for running and visualizing predictions made by our **Prototypical Network** on few-shot tasks using the **ISIC 2019** skin lesion dataset.

---

## 📁 Files Overview

| File                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `demo.py`              | Core CLI demo script for evaluating few-shot predictions from a trained model. |
| `gradio_ui.py`         | Web UI built with Gradio for live interaction with the few-shot classifier.   |
| `data_augment_demo.py` | Displays advanced augmentation effects used in training/evaluation.           |
| `visualizer.py`        | Utility functions for visualizing support/query sets and predictions.        |
| `demo_config.yaml`     | Sample configuration for demo inference (model path, data path, few-shot parameters). |

---

## ⚙️ Setup

Make sure you've installed all required dependencies listed in the root `requirements.txt` file.  
Also, ensure your trained model checkpoint exists at the specified location.

Example:

models/protonet_best.pth



---

## 🚀 Running the Demo (CLI)

You can run the main demo directly via the command line:

```bash
python demo.py \
  --model_path models/protonet_best.pth \
  --test_path data/test \
  --n_way 5 \
  --k_shot 5 \
  --query 5

Or 

using a YAML config:

python demo.py --config demo/demo_config.yaml
The CLI will output few-shot classification accuracy and optionally show support/query visualizations.

🌐 Launching the Gradio Interface
To use the interactive web UI:


python gradio_ui.py
This will launch a local Gradio app where you can:

Upload a lesion image

Visualize the support set and prediction

View class-wise confidence scores

🧪 Visualizing Data Augmentation
To preview the augmentations applied during training, run:


python data_augment_demo.py --image_dir data/test/NV --num_images 5
This displays original and augmented image pairs side-by-side using advanced transforms from utils/augmentation.py.

🖼 Sample Output
✅ Terminal:


[INFO] Accuracy on few-shot test set: 91.20%
🧠 Visualizer (query with prediction labels):


Query 1: MEL | Pred: MEL  
Query 2: BKL | Pred: NV  
...
🌐 Gradio UI:


Upload → See Result → Compare with Support Set