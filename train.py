import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from data_loader import setup_data_loaders, IMAGE_SIZE, OUTPUT_DIR
import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression

# --- Configuration ---
EPOCHS = 20
LEARNING_RATE = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

# --- Model Definition ---
def get_model(model_name, num_classes=1, weights_path=None):
    weights = None
    if weights_path is None:
        if model_name == 'EfficientNet':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
            for param in model.features[8].parameters():
                param.requires_grad = True

        elif model_name == 'ResNet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            model = models.resnet50(weights=weights)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            for param in model.layer4.parameters():
                param.requires_grad = True

        elif model_name == 'InceptionV3':
            weights = models.Inception_V3_Weights.IMAGENET1K_V1
            model = models.inception_v3(weights=weights, aux_logits=True)
            model.aux_logits = False
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            for param in model.Mixed_7c.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown model: {model_name}")

        for param in model.parameters():
            param.requires_grad = False

        if model_name == 'EfficientNet':
             for param in model.classifier.parameters(): param.requires_grad = True
             for param in model.features[8].parameters(): param.requires_grad = True
        elif model_name == 'ResNet50':
            for param in model.fc.parameters(): param.requires_grad = True
            for param in model.layer4.parameters(): param.requires_grad = True
        elif model_name == 'InceptionV3':
             for param in model.Mixed_7c.parameters(): param.requires_grad = True
             for param in model.fc.parameters(): param.requires_grad = True

    else:
        if model_name == 'EfficientNet':
            model = models.efficientnet_b0(weights=None)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)
        elif model_name == 'ResNet50':
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'InceptionV3':
            model = models.inception_v3(weights=None, aux_logits=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            model.aux_logits = False

        model.load_state_dict(torch.load(weights_path, map_location=device))

    return model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, model_name, epochs):
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': [],
               'precision': [], 'val_precision': [], 'recall': [], 'val_recall': [], 'lr': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_soft_preds = np.array([])
    best_labels = np.array([])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['loss'].append(epoch_loss)

        model.eval()
        val_loss = 0.0
        all_preds, epoch_soft_preds, epoch_labels = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple): outputs = outputs[0]
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                soft_preds = torch.sigmoid(outputs)
                epoch_soft_preds.append(soft_preds.cpu().numpy())
                all_preds.append((soft_preds > 0.5).float().cpu().numpy())
                epoch_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        history['lr'].append(current_lr)

        all_preds = np.vstack(all_preds).flatten()
        all_labels_np = np.vstack(epoch_labels).flatten()
        val_acc = accuracy_score(all_labels_np, all_preds)
        val_prec = precision_score(all_labels_np, all_preds, zero_division=0)
        val_rec = recall_score(all_labels_np, all_preds, zero_division=0)

        history['accuracy'].append(np.nan)
        history['precision'].append(np.nan)
        history['recall'].append(np.nan)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Rec: {val_rec:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_soft_preds = np.vstack(epoch_soft_preds).flatten()
            best_labels = all_labels_np
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'models', f'{model_name}_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history, best_val_acc, best_soft_preds, best_labels

def get_soft_predictions(model_name, test_loader):
    """
    Generates predictions WITHOUT Test-Time Augmentation (TTA).
    """
    weights_path = os.path.join(OUTPUT_DIR, 'models', f'{model_name}_final.pth')
    model = get_model(model_name, weights_path=weights_path)
    model.eval()
    all_soft_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple): outputs = outputs[0]
            probs = torch.sigmoid(outputs)
            all_soft_preds.append(probs.cpu().numpy())

    return np.vstack(all_soft_preds).flatten()

def calculate_metrics(true_labels, predictions, method_name):
    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    print(f"\n--- {method_name} Test Results ---")
    print(f"Accuracy: {acc * 100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f}")
    return acc, prec, rec

def evaluate_ensemble(trained_model_names, all_val_data):
    print("\n--- Starting Ensemble Evaluation ---")
    all_model_predictions = {}
    all_test_soft_preds = {}

    # Base loader for labels
    _, _, test_loader_base, _, hf = setup_data_loaders(resize_dim=299)
    true_labels = []
    for _, labels in test_loader_base: true_labels.append(labels.numpy())
    true_labels = np.vstack(true_labels).flatten()
    hf.close()

    for name in trained_model_names:
        resize_dim = 299 if name == 'InceptionV3' else 224
        print(f"Generating predictions for {name} ({resize_dim}x{resize_dim})...")
        _, _, test_loader_current, _, current_hf = setup_data_loaders(resize_dim=resize_dim)

        test_soft = get_soft_predictions(name, test_loader_current)
        all_test_soft_preds[name] = test_soft
        all_model_predictions[name] = (test_soft > 0.5).astype(float)
        current_hf.close()

    min_len = len(true_labels)
    effnet_hard = all_model_predictions['EfficientNet'][:min_len]
    resnet_hard = all_model_predictions['ResNet50'][:min_len]
    inception_hard = all_model_predictions['InceptionV3'][:min_len]

    # 1. Max Voting
    stacked = np.stack([effnet_hard, resnet_hard, inception_hard], axis=1)
    voting_preds, _ = mode(stacked, axis=1, keepdims=False)
    calculate_metrics(true_labels, voting_preds, "Max Voting")

    # 2. Weighted Avg
    val_accs = {k: v['acc'] for k, v in all_val_data.items()}
    total_acc = sum(val_accs.values())
    weights = {k: v / total_acc for k, v in val_accs.items()}
    print(f"\nWeights: {weights}")

    weighted_soft = (
        weights['EfficientNet'] * all_test_soft_preds['EfficientNet'][:min_len] +
        weights['ResNet50'] * all_test_soft_preds['ResNet50'][:min_len] +
        weights['InceptionV3'] * all_test_soft_preds['InceptionV3'][:min_len]
    )
    calculate_metrics(true_labels, (weighted_soft > 0.5).astype(float), "Weighted Avg")

    # 3. Stacking
    X_val = np.stack([
        all_val_data['EfficientNet']['soft_preds'],
        all_val_data['ResNet50']['soft_preds'],
        all_val_data['InceptionV3']['soft_preds']
    ], axis=1)
    y_val = all_val_data['EfficientNet']['labels']

    meta = LogisticRegression(solver='liblinear', random_state=42)
    meta.fit(X_val, y_val)

    X_test = np.stack([
        all_test_soft_preds['EfficientNet'][:min_len],
        all_test_soft_preds['ResNet50'][:min_len],
        all_test_soft_preds['InceptionV3'][:min_len]
    ], axis=1)
    stack_preds = meta.predict(X_test)
    calculate_metrics(true_labels, stack_preds, "Stacking")

def run_training_workflow():
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'history'), exist_ok=True)

    models_to_train = ['InceptionV3', 'EfficientNet', 'ResNet50']
    HDF5_FILE_HANDLE = None
    trained_model_names = []
    all_val_data = {}

    for name in models_to_train:
        print(f"\n--- Training {name} ---")
        resize_dim = 299 if name == 'InceptionV3' else 224

        train_loader, val_loader, _, pos_weight, current_hdf5_file = setup_data_loaders(resize_dim=resize_dim)
        if HDF5_FILE_HANDLE is None: HDF5_FILE_HANDLE = current_hdf5_file

        model = get_model(name)
        criterion = FocalLoss(alpha=pos_weight.to(device), gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

        history_data, val_acc, best_val_soft, best_val_labels = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler, name, EPOCHS
        )

        pd.DataFrame(history_data).to_csv(os.path.join(OUTPUT_DIR, 'history', f'{name}_history.csv'), index=False)
        pd.DataFrame({'Model': [name], 'Input_Shape': [f'{resize_dim}x{resize_dim}'], 'Epochs': [len(history_data['loss'])]}).to_csv(os.path.join(OUTPUT_DIR, 'history', f'{name}_params.csv'), index=False)

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'models', f'{name}_final.pth'))
        trained_model_names.append(name)
        all_val_data[name] = {'acc': val_acc, 'soft_preds': best_val_soft, 'labels': best_val_labels}

    if trained_model_names: evaluate_ensemble(trained_model_names, all_val_data)
    if HDF5_FILE_HANDLE: HDF5_FILE_HANDLE.close()
    print("\nWorkflow Complete.")

if __name__ == '__main__':
    run_training_workflow()
