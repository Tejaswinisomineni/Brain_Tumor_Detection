"""
CNN + LSTM Hybrid Model for Brain Tumor Detection
===================================================
This script implements a mixed CNN+LSTM architecture for brain tumor classification.

Architecture Overview:
- CNN Backbone (ResNet18): Extracts spatial features from MRI images
- LSTM Layer: Processes the CNN feature sequence for temporal/sequential pattern recognition
- Fully Connected Classifier: Final classification (Tumor vs Healthy)

The CNN extracts spatial features from image patches/regions, and the LSTM
captures dependencies between these features for improved classification.
"""

# ============================================================
# 1. Import Essential Libraries
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='darkgrid')
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, random_split
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchsummary import summary
from sklearn.metrics import confusion_matrix, classification_report
import itertools
from tqdm import tqdm
import pathlib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CNN + LSTM Brain Tumor Detection")
print("=" * 60)

# ============================================================
# 2. CNN + LSTM Hybrid Model Definition
# ============================================================

class CNN_LSTM_Model(nn.Module):
    """
    Hybrid CNN + LSTM Model for Brain Tumor Classification.
    
    Architecture:
    1. CNN Feature Extractor (ResNet18 pretrained):
       - Extracts rich spatial features from input images
       - The final classification layer is removed
       - Output: 512-dimensional feature vector
    
    2. LSTM Sequence Processor:
       - Takes CNN features and processes them as a sequence
       - The image features are reshaped into a sequence of patches
       - Captures long-range dependencies in the feature space
    
    3. Fully Connected Classifier:
       - Takes LSTM output and produces final classification
       - Dropout for regularization
       - Output: num_classes (2 for tumor/healthy)
    """
    
    def __init__(self, num_classes=2, hidden_size=256, num_layers=2, 
                 dropout=0.3, bidirectional=True):
        super(CNN_LSTM_Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # ---- CNN Backbone (ResNet18) ----
        # Use pretrained ResNet18 as feature extractor
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        # This gives us a 512-dim feature vector per image
        self.cnn_features = nn.Sequential(*list(resnet.children())[:-2])
        # Output shape after CNN: (batch, 512, H', W') where H'=W'=8 for 256x256 input
        
        # Freeze early CNN layers (optional - for transfer learning)
        for param in list(self.cnn_features.parameters())[:20]:
            param.requires_grad = False
        
        # ---- Attention-enhanced feature processing ----
        # Adaptive pooling to get fixed-size feature maps
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        # This creates 4x4=16 spatial positions, each with 512 features
        
        # ---- LSTM Layer ----
        # Input: sequence of spatial features from CNN
        # Each spatial position becomes a time step
        # Feature dim = 512 (ResNet18 output channels)
        self.lstm = nn.LSTM(
            input_size=512,       # Feature dimension from CNN
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # ---- Attention Mechanism ----
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # ---- Classifier Head ----
        lstm_output_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x, extract_features=False):
        batch_size = x.size(0)
        
        # ---- Step 1: CNN Feature Extraction ----
        # x shape: (batch, 3, 256, 256)
        cnn_out = self.cnn_features(x)
        # cnn_out shape: (batch, 512, 8, 8)
        
        # ---- Step 2: Reshape for LSTM ----
        # Apply adaptive pooling to get fixed spatial grid
        pooled = self.adaptive_pool(cnn_out)
        # pooled shape: (batch, 512, 4, 4)
        
        # Reshape: treat each spatial position as a time step
        # (batch, channels, h, w) -> (batch, h*w, channels)
        b, c, h, w = pooled.size()
        seq_len = h * w  # 16 time steps
        lstm_input = pooled.view(b, c, seq_len)  # (batch, 512, 16)
        lstm_input = lstm_input.permute(0, 2, 1)  # (batch, 16, 512)
        
        if extract_features:
            return lstm_input
        
        # ---- Step 3: LSTM Processing ----
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * self.num_directions, 
                         batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, 
                         batch_size, self.hidden_size).to(x.device)
        
        lstm_out, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        # lstm_out shape: (batch, 16, hidden_size * num_directions)
        
        # ---- Step 4: Attention-weighted aggregation ----
        attention_weights = self.attention(lstm_out)  # (batch, 16, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch, 16, 1)
        
        # Weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size * 2)
        
        # ---- Step 5: Classification ----
        output = self.classifier(context)
        
        return output


# ============================================================
# 3. Also define the original CNN model for comparison
# ============================================================

class CNN_TUMOR(nn.Module):
    """Original CNN Model (for comparison)"""
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
    
    def forward(self, x, extract_features=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, self.num_flatten)
        
        if extract_features:
            return x
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


# ============================================================
# 4. Feature Extraction Model (ResNet18 backbone)
# ============================================================

class FeatureExtractor(nn.Module):
    """ResNet18-based Feature Extractor for CNN+LSTM pipeline"""
    def __init__(self, num_classes=2):
        super(FeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove classification layer
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x, extract_features=False):
        features = self.resnet(x)
        if extract_features:
            return features
        return self.fc(features)


# ============================================================
# 5. Data Loading and Preparation
# ============================================================

def prepare_data(data_dir, batch_size=32):
    """Prepare training and validation data loaders"""
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    data_path = pathlib.Path(data_dir)
    
    # Check if split data exists
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    if train_dir.exists() and val_dir.exists():
        train_set = ImageFolder(str(train_dir), transform=transform)
        val_set = ImageFolder(str(val_dir), transform=val_transform)
        print(f"Training samples: {len(train_set)}")
        print(f"Validation samples: {len(val_set)}")
        print(f"Classes: {train_set.classes}")
    else:
        # Use the main dataset directory
        dataset_dir = data_path / "Brain Tumor Data Set"
        if not dataset_dir.exists():
            print(f"Error: Dataset directory not found at {dataset_dir}")
            return None, None, None
        
        full_dataset = ImageFolder(str(dataset_dir), transform=transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Get class labels
    if hasattr(train_set, 'class_to_idx'):
        CLA_label = train_set.class_to_idx
    elif hasattr(train_set, 'dataset'):
        CLA_label = train_set.dataset.class_to_idx
    else:
        CLA_label = {"Brain Tumor": 0, "Healthy": 1}
    
    return train_loader, val_loader, CLA_label


# ============================================================
# 6. Training Functions
# ============================================================

def get_lr(opt):
    """Get current learning rate"""
    for param_group in opt.param_groups:
        return param_group['lr']

def loss_batch(loss_func, output, target, opt=None):
    """Compute loss for a single batch"""
    loss = loss_func(output, target)
    pred = output.argmax(dim=1, keepdim=True)
    metric_b = pred.eq(target.view_as(pred)).sum().item()
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, device, opt=None):
    """Compute loss for one epoch"""
    run_loss = 0.0
    t_metric = 0.0
    len_data = len(dataset_dl.dataset)
    
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        run_loss += loss_b
        if metric_b is not None:
            t_metric += metric_b
    
    loss = run_loss / float(len_data)
    metric = t_metric / float(len_data)
    return loss, metric

def Train_Val(model, params, device, verbose=True):
    """Train and validate the model"""
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]
    
    loss_history = {"train": [], "val": []}
    metric_history = {"train": [], "val": []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training"):
        current_lr = get_lr(opt)
        if verbose:
            print(f'\nEpoch {epoch}/{epochs - 1}, current lr={current_lr}')
        
        # Train
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, device, opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, device)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if verbose:
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        
        if verbose:
            print(f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}%")
            print("-" * 40)
    
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


# ============================================================
# 7. Evaluation Functions
# ============================================================

def True_and_Pred(val_loader, model, device):
    """Get true labels and predictions"""
    y_true = []
    y_pred = []
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.numpy()
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        pred = pred.detach().cpu().numpy()
        y_true = np.append(y_true, labels)
        y_pred = np.append(y_pred, pred)
    return y_true, y_pred

def plot_training_history(loss_hist, metric_hist, epochs, title_prefix=""):
    """Plot training and validation loss/accuracy"""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    sns.lineplot(x=list(range(1, epochs+1)), y=loss_hist["train"], ax=ax[0], label='Train Loss')
    sns.lineplot(x=list(range(1, epochs+1)), y=loss_hist["val"], ax=ax[0], label='Val Loss')
    ax[0].set_title(f'{title_prefix}Loss History')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    
    # Accuracy plot
    sns.lineplot(x=list(range(1, epochs+1)), y=metric_hist["train"], ax=ax[1], label='Train Accuracy')
    sns.lineplot(x=list(range(1, epochs+1)), y=metric_hist["val"], ax=ax[1], label='Val Accuracy')
    ax[1].set_title(f'{title_prefix}Accuracy History')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(f'{title_prefix.strip().replace(" ", "_")}training_history.png', dpi=150)
    plt.show()

def show_confusion_matrix(cm, CLA_label, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 7))
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.YlGnBu)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLA_label))
    
    plt.xticks(tick_marks, [f"{value}={key}" for key, value in CLA_label.items()], rotation=45)
    plt.yticks(tick_marks, [f"{value}={key}" for key, value in CLA_label.items()])
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i,j]}\n{cm[i,j]/np.sum(cm)*100:.2f}%",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150)
    plt.show()


# ============================================================
# 8. Main Execution
# ============================================================

def main():
    # ---- Configuration ----
    BATCH_SIZE = 16  # Smaller batch for CNN+LSTM (more memory intensive)
    EPOCHS = 40
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2
    HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.3
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ---- Data Preparation ----
    print("\n" + "=" * 60)
    print("Loading Data...")
    print("=" * 60)
    
    # Update this path to your data directory
    data_dir = os.path.dirname(os.path.abspath(__file__))
    brain_dir = os.path.join(data_dir, "brain")
    
    train_loader, val_loader, CLA_label = prepare_data(brain_dir, BATCH_SIZE)
    
    if train_loader is None:
        print("Failed to load data. Please check the data directory.")
        return
    
    print(f"Class labels: {CLA_label}")
    
    # ---- Model Creation ----
    print("\n" + "=" * 60)
    print("Creating CNN+LSTM Model...")
    print("=" * 60)
    
    cnn_lstm_model = CNN_LSTM_Model(
        num_classes=NUM_CLASSES,
        hidden_size=HIDDEN_SIZE,
        num_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        bidirectional=True
    ).to(device)
    
    print("\n--- CNN+LSTM Model Architecture ---")
    print(cnn_lstm_model)
    
    # Count parameters
    total_params = sum(p.numel() for p in cnn_lstm_model.parameters())
    trainable_params = sum(p.numel() for p in cnn_lstm_model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ---- Training ----
    print("\n" + "=" * 60)
    print("Training CNN+LSTM Model...")
    print("=" * 60)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, cnn_lstm_model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    params_train = {
        "train": train_loader,
        "val": val_loader,
        "epochs": EPOCHS,
        "optimiser": optimizer,
        "lr_change": scheduler,
        "f_loss": nn.NLLLoss(reduction="sum"),
        "weight_path": os.path.join(data_dir, "cnn_lstm_weights.pt"),
    }
    
    cnn_lstm_model, loss_hist, metric_hist = Train_Val(
        cnn_lstm_model, params_train, device, verbose=True
    )
    
    # ---- Plot Training History ----
    plot_training_history(loss_hist, metric_hist, EPOCHS, "CNN_LSTM_")
    
    # ---- Evaluation ----
    print("\n" + "=" * 60)
    print("Evaluating CNN+LSTM Model...")
    print("=" * 60)
    
    cnn_lstm_model.eval()
    y_true, y_pred = True_and_Pred(val_loader, cnn_lstm_model, device)
    
    print("\n--- Classification Report (CNN+LSTM) ---")
    print(classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred)
    show_confusion_matrix(cm, CLA_label, title='CNN+LSTM Confusion Matrix')
    
    # ---- Save Full Model ----
    model_path = os.path.join(data_dir, "cnn_lstm_brain_tumor_model.pt")
    torch.save(cnn_lstm_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # ---- Test with sample image ----
    print("\n" + "=" * 60)
    print("Testing with sample images...")
    print("=" * 60)
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Inverse class label mapping
    inv_CLA_label = {v: k for k, v in CLA_label.items()}
    
    test_images = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.startswith("test_data") and f.endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if test_images:
        cnn_lstm_model.eval()
        fig, axes = plt.subplots(1, min(len(test_images), 5), figsize=(20, 4))
        if not hasattr(axes, '__len__'):
            axes = [axes]
        
        for idx, img_path in enumerate(test_images[:5]):
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = test_transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = cnn_lstm_model(img_tensor)
                    _, predicted = torch.max(output, 1)
                    pred_class = inv_CLA_label.get(predicted.item(), "Unknown")
                
                axes[idx].imshow(np.array(Image.open(img_path).resize((256, 256))))
                axes[idx].set_title(f"Pred: {pred_class}")
                axes[idx].axis('off')
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "cnn_lstm_predictions.png"), dpi=150)
        plt.show()
    
    print("\n" + "=" * 60)
    print("CNN+LSTM Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
