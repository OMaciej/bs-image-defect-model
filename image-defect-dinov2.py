import os
import pandas as pd
import warnings

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torchcodec.decoders import decode_image
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import GroupKFold

from tqdm import tqdm

# Constants
DEBUG = True

CSV_PATH = 'data/splatdata.csv'
IMG_DIR_PATH = 'data/images/'
FINAL_MODEL_SAVE_DIR ='model'
FINAL_MODEL_SAVE_NAME = 'baseline.json'

DINO_REPO = 'facebookresearch/dinov2'
DINO_MODEL_NAME = 'dinov2_vits14'

DINO_IMG_HEIGHT = 420
DINO_IMG_WIDTH = 560

LOADER_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 128

EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

EPSILON = 1e-6

LABELS = ['edge_fraction', 'visgauss_fraction', 
          'blur_fraction', 'unreal_fraction']

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')

class ImageDataset(Dataset):
    def __init__(self, img_paths: list[str]):
        self.img_paths = img_paths
        self.resize_transform = transforms.Resize((DINO_IMG_HEIGHT, DINO_IMG_WIDTH))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        item_path = self.img_paths[index]

        img = decode_image(item_path)

        # In order for DINOv2 to be able to calculate our image's features,
        # its width and height must be divisible by 14. Therefore, we resize
        # our image.
        img = self.resize_transform(img)

        return img

# MLP network that takes 1152 image features (calculated by DINOv2) as input
# and outputs 4 predictions.
class DefectClassifier(nn.Module):
    def __init__(self, in_features: int = 1152):
        super().__init__()

        self.net = nn.Sequential(
            # Prepare input
            nn.LayerNorm(in_features), # normalize features
            nn.Dropout(0.3), # randomly disable 30% of all features

            # First hidden layer
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            # Second hidden layer
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),

            # Output
            nn.Linear(64, 4)
        )

    def forward(self, x) -> torch.Tensor:
        return self.net(x)


# Extracts features from images using DINOv2.
@torch.no_grad()
def extract_features(df, device) -> torch.Tensor:
    paths = [
        os.path.join(IMG_DIR_PATH, img_set, img_id)
        for img_set, img_id in df[['set', 'id']].itertuples(index=False)
    ]

    dataset = ImageDataset(paths)
    is_device_cuda = (device.type == 'cuda')
    data_loader = DataLoader(
        dataset,
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=is_device_cuda
    )

    dino = torch.hub.load(DINO_REPO, DINO_MODEL_NAME)
    dino = dino.to(device)
    dino.eval()

    # Mean and standard deviation of pixel brightness for each color channel
    # in the ImageNet dataset. Used to standardize our images.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, 
                        dtype=torch.float32).view(1, 3, 1, 1)

    std = torch.tensor([0.229, 0.224, 0.225], device=device, 
                       dtype=torch.float32).view(1, 3, 1, 1)

    extracted_features = []

    for batch in tqdm(data_loader, total=len(data_loader)):
        # We move our image to the GPU and standardize it.
        batch_dev = batch.to(device, non_blocking=is_device_cuda).float().div_(255.0)
        batch_dev = (batch_dev - mean) / std

        # We calculate features using DINOv2.
        feature_dict = dino.forward_features(batch_dev)
        cls_token = feature_dict['x_norm_clstoken']
        patch_tokens = feature_dict['x_norm_patchtokens']

        mean_patch = patch_tokens.mean(dim=1)
        max_patch = patch_tokens.max(dim=1).values

        combined_features = torch.cat([cls_token, mean_patch, max_patch], dim=1)
        extracted_features.append(combined_features.float())

    del dino

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return torch.cat(extracted_features)

def train_model(
        x_train : torch.Tensor, 
        y_train : torch.Tensor, 
        device : torch.device
    ) -> nn.Module:

    model = DefectClassifier(in_features=x_train.shape[1]).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=False)

            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()

            optimizer.step()

        scheduler.step()

    model.eval()

    return model

@torch.no_grad()
def predict(model : nn.Module, x : torch.Tensor) -> torch.Tensor:
    model.eval()
    logits = model(x)

    return torch.sigmoid(logits)

@torch.no_grad()
def find_best_f1_thresholds(y_true_bin : torch.Tensor, y_pred : torch.Tensor) -> torch.Tensor:
    candidate_thresholds = torch.linspace(0.2, 0.7, 51, device=y_pred.device)
    preds_bin = (y_pred.unsqueeze(0) >= candidate_thresholds.view(51, 1, 1)).float()
    trues_bin = y_true_bin.unsqueeze(0)

    tp = ((trues_bin == 1) & (preds_bin == 1)).sum(dim=1)
    fp = ((trues_bin == 0) & (preds_bin == 1)).sum(dim=1)
    fn = ((trues_bin == 1) & (preds_bin == 0)).sum(dim=1)

    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)

    f1_matrix = (2.0 * precision * recall) / (precision + recall + EPSILON)

    best_indices = torch.argmax(f1_matrix, dim=0)
    return candidate_thresholds[best_indices] 

@torch.no_grad()
def f1_score_per_label(y_true_bin, y_pred_bin) -> torch.Tensor:
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum(dim=0)
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum(dim=0)
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum(dim=0)

    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)

    return (2.0 * precision * recall) / (precision + recall + EPSILON)

@torch.no_grad()
def log_loss(y_true, y_pred) -> float:
    y_pred_clipped = torch.clip(y_pred, EPSILON, 1 - EPSILON)

    bce = -torch.mean(y_true * torch.log(y_pred_clipped) + 
                       (1 - y_true) * torch.log((1 - y_pred_clipped)))

    return float(bce.item())

if __name__ == '__main__':
    device = get_device()

    if device.type != 'cuda':
        print('GPU not found. Running on CPU.\n')

    df = pd.read_csv(CSV_PATH, nrows=256)    

    if not DEBUG:
        print('Extracting features using DINOv2.')
        x = extract_features(df, device)
        y = torch.tensor(df[LABELS].values, dtype=torch.float32, device=device)

        splitter = GroupKFold(n_splits=2)
        groups = df['set'].values

        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups)):
            print(f'\nFOLD {fold+1}/{splitter.get_n_splits()}')

            train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
            test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]

            model = train_model(x_train, y_train, device)

            y_pred = predict(model, x_test)

            bce = log_loss(y_test, y_pred)
            print(f'BCE: {bce}')
    else:
        x_train_dummy, y_train_dummy = torch.randn(100, 1152, device=device), torch.rand(100, 4, device=device)
        x_test_dummy, y_test_dummy = torch.randn(25, 1152, device=device), torch.rand(25, 4, device=device)
        y_test_bin_dummy = (y_test_dummy >= 0.5).float()
        y_train_bin_dummy = (y_train_dummy >= 0.5).float()

        model = train_model(x_train_dummy, y_train_dummy, device)

        y_pred_test_dummy = predict(model, x_test_dummy)
        y_pred_train_dummy = predict(model, x_train_dummy)

        bce = log_loss(y_test_dummy, y_pred_test_dummy)

        f1_thresholds = find_best_f1_thresholds(y_train_bin_dummy, y_pred_train_dummy)

        y_pred_bin_dummy = (y_pred_test_dummy >= f1_thresholds).float()

        f1_per_label = f1_score_per_label(y_test_bin_dummy, y_pred_bin_dummy)
        f1_mean = f1_per_label.mean()

        print(f'F1: {f1_mean} | BCE: {bce}')