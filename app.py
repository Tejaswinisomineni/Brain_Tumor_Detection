"""
Brain Tumor Detection — Full-Featured Web Application
=======================================================
Complete Flask backend serving CNN+LSTM model with all features:
  - Single image classification
  - Batch prediction
  - Training history visualization
  - Model architecture info
  - Dataset statistics
  - Model comparison (CNN vs CNN+LSTM)
"""

import os
import io
import base64
import json
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Model Definitions
# ============================================================

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2,
                 dropout=0.3, bidirectional=True):
        super(CNN_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        resnet = models.resnet18(pretrained=False)
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-2])
        for param in list(self.cnn_features.parameters())[:20]:
            param.requires_grad = False
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(), nn.Linear(hidden_size, 1))
        lstm_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes), nn.LogSoftmax(dim=1))

    def forward(self, x):
        batch_size = x.size(0)
        cnn_out = self.cnn_features(x)
        pooled = self.adaptive_pool(cnn_out)
        b, c, h, w = pooled.size()
        lstm_input = pooled.view(b, c, h * w).permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        attn = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn, dim=1)
        return self.classifier(context)


class CNN_TUMOR(nn.Module):
    def __init__(self, params):
        super(CNN_TUMOR, self).__init__()
        C_in, H_in, W_in = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]
        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        self.num_flatten = self._get_flatten_size(C_in, H_in, W_in)
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def _get_flatten_size(self, C_in, H_in, W_in):
        x = torch.zeros(1, C_in, H_in, W_in)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, self.num_flatten)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ============================================================
# Flask App
# ============================================================

app = Flask(__name__, static_folder='static')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = {0: "Brain Tumor", 1: "Healthy"}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global model holders
models_loaded = {}

def load_all_models():
    """Load all available models"""
    global models_loaded

    # CNN model
    cnn_path = os.path.join(BASE_DIR, "Brain_Tumor_model.pt")
    if os.path.exists(cnn_path):
        try:
            m = torch.load(cnn_path, map_location=DEVICE, weights_only=False)
            m.eval()
            models_loaded["CNN"] = m
            print(f"  [OK] CNN model loaded")
        except Exception as e:
            print(f"  [FAIL] CNN model failed: {e}")

    # CNN+LSTM full model
    lstm_path = os.path.join(BASE_DIR, "cnn_lstm_brain_tumor_model.pt")
    if os.path.exists(lstm_path):
        try:
            m = torch.load(lstm_path, map_location=DEVICE, weights_only=False)
            m.eval()
            models_loaded["CNN+LSTM"] = m
            print(f"  [OK] CNN+LSTM model loaded")
        except Exception as e:
            print(f"  [FAIL] CNN+LSTM full model failed: {e}")

    # CNN+LSTM weights only
    if "CNN+LSTM" not in models_loaded:
        weights_path = os.path.join(BASE_DIR, "cnn_lstm_weights.pt")
        if os.path.exists(weights_path):
            try:
                m = CNN_LSTM_Model(num_classes=2).to(DEVICE)
                m.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
                m.eval()
                models_loaded["CNN+LSTM"] = m
                print(f"  [OK] CNN+LSTM model loaded (from weights)")
            except Exception as e:
                print(f"  [FAIL] CNN+LSTM weights failed: {e}")


def predict_single(model, image_bytes):
    """Predict a single image"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.exp(output).squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}
    }


def get_dataset_stats():
    """Get dataset statistics"""
    stats = {"train": {}, "val": {}, "total_train": 0, "total_val": 0, "sample_images": []}
    brain_dir = os.path.join(BASE_DIR, "brain")

    for split in ["train", "val"]:
        split_dir = os.path.join(brain_dir, split)
        if os.path.exists(split_dir):
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.bmp'))])
                    stats[split][cls] = count
                    stats[f"total_{split}"] += count

                    # Grab sample images
                    if len(stats["sample_images"]) < 8:
                        imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                        for img_name in imgs[:2]:
                            try:
                                img_path = os.path.join(cls_dir, img_name)
                                img = Image.open(img_path).convert("RGB").resize((128, 128))
                                buf = io.BytesIO()
                                img.save(buf, format='JPEG', quality=70)
                                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                stats["sample_images"].append({
                                    "data": f"data:image/jpeg;base64,{b64}",
                                    "label": cls, "split": split, "name": img_name
                                })
                            except:
                                pass
    return stats


def get_model_architecture():
    """Get model architecture details for display"""
    archs = {}

    if "CNN" in models_loaded:
        m = models_loaded["CNN"]
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        layers = []
        for name, module in m.named_modules():
            if name:
                layers.append({"name": name, "type": module.__class__.__name__,
                               "details": str(module).split("(", 1)[-1].rstrip(")") if "(" in str(module) else ""})
        archs["CNN"] = {
            "total_params": f"{total:,}",
            "trainable_params": f"{trainable:,}",
            "layers": layers[:20],
            "description": "Custom 4-layer CNN with Conv2d → MaxPool2d blocks, followed by fully connected classifier."
        }

    if "CNN+LSTM" in models_loaded:
        m = models_loaded["CNN+LSTM"]
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        archs["CNN+LSTM"] = {
            "total_params": f"{total:,}",
            "trainable_params": f"{trainable:,}",
            "layers": [
                {"name": "cnn_features", "type": "ResNet18 Backbone", "details": "Pretrained, final FC removed"},
                {"name": "adaptive_pool", "type": "AdaptiveAvgPool2d", "details": "Output: 4×4 spatial grid"},
                {"name": "lstm", "type": "LSTM", "details": "512→256, 2 layers, bidirectional"},
                {"name": "attention", "type": "Attention", "details": "512→256→1, Tanh activation"},
                {"name": "classifier.0", "type": "Linear", "details": "512→128"},
                {"name": "classifier.1", "type": "ReLU", "details": ""},
                {"name": "classifier.2", "type": "Dropout", "details": "p=0.3"},
                {"name": "classifier.3", "type": "Linear", "details": "128→64"},
                {"name": "classifier.4", "type": "ReLU", "details": ""},
                {"name": "classifier.5", "type": "Dropout", "details": "p=0.3"},
                {"name": "classifier.6", "type": "Linear", "details": "64→2"},
                {"name": "classifier.7", "type": "LogSoftmax", "details": "dim=1"},
            ],
            "description": "Hybrid CNN+LSTM: ResNet18 extracts spatial features → reshaped to 16-step sequence → Bidirectional LSTM with attention → FC classifier."
        }

    return archs


# ============================================================
# API Routes
# ============================================================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Classify a single image with selected model"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    model_name = request.form.get('model', None)

    # Pick model
    if model_name and model_name in models_loaded:
        model = models_loaded[model_name]
    elif "CNN+LSTM" in models_loaded:
        model = models_loaded["CNN+LSTM"]
        model_name = "CNN+LSTM"
    elif "CNN" in models_loaded:
        model = models_loaded["CNN"]
        model_name = "CNN"
    else:
        return jsonify({"error": "No model loaded"}), 500

    try:
        result = predict_single(model, file.read())
        result["model_type"] = model_name
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict-compare', methods=['POST'])
def api_predict_compare():
    """Classify image with ALL loaded models for comparison"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    image_bytes = file.read()

    results = {}
    for name, model in models_loaded.items():
        try:
            r = predict_single(model, image_bytes)
            r["model_type"] = name
            results[name] = r
        except Exception as e:
            results[name] = {"error": str(e), "model_type": name}

    return jsonify({"comparisons": results, "models_count": len(results)})


@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """Classify multiple images"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    model_name = request.form.get('model', None)
    if model_name and model_name in models_loaded:
        model = models_loaded[model_name]
    elif "CNN+LSTM" in models_loaded:
        model = models_loaded["CNN+LSTM"]
        model_name = "CNN+LSTM"
    elif "CNN" in models_loaded:
        model = models_loaded["CNN"]
        model_name = "CNN"
    else:
        return jsonify({"error": "No model loaded"}), 500

    results = []
    tensors = []
    filenames = []
    
    # Preprocess all images
    for f in files[:50]:  # Limit increased to 50
        try:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
            img_tensor = transform(img)
            tensors.append(img_tensor)
            filenames.append(f.filename)
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    # Run batched inference
    if tensors:
        batch_tensor = torch.stack(tensors).to(DEVICE)
        model.eval()
        with torch.no_grad():
            output = model(batch_tensor)
            probs = torch.exp(output).cpu().numpy()
            
        for i in range(len(tensors)):
            pred_idx = int(np.argmax(probs[i]))
            results.append({
                "filename": filenames[i],
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": round(float(probs[i][pred_idx]) * 100, 2),
                "probabilities": {CLASS_NAMES[j]: round(float(probs[i][j]) * 100, 2) for j in range(len(CLASS_NAMES))}
            })

    # Summary
    tumor_count = sum(1 for r in results if r.get("prediction") == "Brain Tumor")
    healthy_count = sum(1 for r in results if r.get("prediction") == "Healthy")

    return jsonify({
        "results": results,
        "summary": {"total": len(results), "tumor": tumor_count, "healthy": healthy_count},
        "model_type": model_name
    })


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Return info about loaded models"""
    return jsonify({
        "models": list(models_loaded.keys()),
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "input_size": "256×256",
        "status": "ready" if models_loaded else "not_ready"
    })


@app.route('/api/model-architecture', methods=['GET'])
def api_model_architecture():
    """Return detailed model architecture"""
    return jsonify(get_model_architecture())


@app.route('/api/dataset-stats', methods=['GET'])
def api_dataset_stats():
    """Return dataset statistics and sample images"""
    return jsonify(get_dataset_stats())


@app.route('/api/training-history', methods=['GET'])
def api_training_history():
    """Return training history if saved plots exist"""
    plots = {}
    for name in ["CNN_LSTM_training_history.png", "training_history.png"]:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                plots[name.replace(".png", "")] = f"data:image/png;base64,{b64}"

    # Check for confusion matrix
    for name in ["CNN+LSTM_Confusion_Matrix.png", "Confusion_Matrix.png"]:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                plots[name.replace(".png", "")] = f"data:image/png;base64,{b64}"

    return jsonify({"plots": plots, "available": len(plots) > 0})


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  Brain Tumor Detection — Full Web Application")
    print("=" * 60)
    print("\nLoading models...")
    load_all_models()
    print(f"\nModels ready: {list(models_loaded.keys())}")
    print(f"Device: {DEVICE}")
    
    # Use PORT from environment for Render/deployment compatibility
    port = int(os.environ.get("PORT", 5000))
    print(f"\n-> Starting server on port {port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
