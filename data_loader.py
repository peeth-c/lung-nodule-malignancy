import h5py
import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Configuration ---
DATA_DIR = 'lung-nodule-malignancy'
OUTPUT_DIR = 'output_data'
HDF5_DATASET_KEY = 'ct_slices'
IMAGE_SIZE = 64
HDF5_PATH = os.path.join(DATA_DIR, 'all_patches.hdf5')
CSV_PATH = os.path.join(DATA_DIR, 'malignancy.csv')
RANDOM_STATE = 42
BATCH_SIZE = 32

# --- HDF5 PyTorch Dataset ---
class HDF5Dataset(Dataset):
    """PyTorch Dataset for loading 1-channel patches from HDF5."""
    def __init__(self, data_ref, indices, labels, resize_dim=None, is_train=False):
        self.data_ref = data_ref
        self.indices = indices
        self.labels = labels

        # Standard normalization values for ImageNet-pretrained models
        NORM_MEAN = [0.485, 0.456, 0.406]
        NORM_STD = [0.229, 0.224, 0.225]

        transform_list = []

        # 1. Convert NumPy array (HWC) to Tensor (CHW) first
        transform_list.append(transforms.ToTensor())

        # --- REVERTED: Augmentations Removed ---
        # Note: Previous Rotational, Flip, and Affine transforms are deleted.

        # 2. Conditional Resize
        if resize_dim is not None and resize_dim != IMAGE_SIZE:
             transform_list.append(transforms.Resize((resize_dim, resize_dim), antialias=True))

        # 3. Repeat 1 channel to 3 channels (Required for Transfer Learning)
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))

        # 4. Normalization (CRITICAL for transfer learning)
        transform_list.append(transforms.Normalize(mean=NORM_MEAN, std=NORM_STD))

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        hdf5_idx = self.indices[idx]
        image = self.data_ref[hdf5_idx]
        label = self.labels[hdf5_idx]

        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return image, label

def setup_data_loaders(resize_dim=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_labels = pd.read_csv(CSV_PATH)
    labels = df_labels['malignancy'].values

    # Open HDF5 file
    HDF5_FILE = h5py.File(HDF5_PATH, 'r')
    DATA_REF = HDF5_FILE[HDF5_DATASET_KEY]

    indices = np.arange(len(df_labels))

    # 1. Split into 80% Training and 20% Temp (Validation + Test)
    train_indices, temp_indices, _, _ = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )

    # 2. Split Temp into 50% Validation and 50% Test (10% each of total)
    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, labels[temp_indices], test_size=0.5, stratify=labels[temp_indices], random_state=RANDOM_STATE
    )

    # Recalculate class weights using only the new training set
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels[train_indices]
    )
    # Weight for the positive class (1.0)
    pos_weight = torch.tensor(class_weights[1] / class_weights[0], dtype=torch.float32)

    # Create Datasets
    train_dataset = HDF5Dataset(DATA_REF, train_indices, labels, resize_dim=resize_dim, is_train=True)
    val_dataset = HDF5Dataset(DATA_REF, val_indices, labels, resize_dim=resize_dim, is_train=False)
    test_dataset = HDF5Dataset(DATA_REF, test_indices, labels, resize_dim=resize_dim, is_train=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    print(f"Data loading set up. Train samples: {len(train_indices)}, Val samples: {len(val_indices)}, Test samples: {len(test_indices)}. Input Size: {resize_dim if resize_dim else IMAGE_SIZE}x{resize_dim if resize_dim else IMAGE_SIZE}.")

    return train_loader, val_loader, test_loader, pos_weight, HDF5_FILE

if __name__ == '__main__':
    train_loader, val_loader, test_loader, pos_weight, hf = setup_data_loaders()
    hf.close()
    print("Data setup complete and HDF5 file closed.")
