import os
import cv2
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

# Constants
CSV_PATH = 'data/partialdata.csv'
IMG_DIR_PATH = 'data/images/'
LABELS = ['edge_fraction', 'visgauss_fraction', 'blur_fraction', 'unreal_fraction']
EPSILON = 1e-6

def compute_image_features(img_path):
    img = cv2.imread(img_path)

    height, width, _ = np.shape(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Features for determining blur.

    # We calculate the blurrines using the laplacian operator.
    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    lap_abs = np.abs(lap)

    laplacian_var = float(np.var(lap))
    laplacian_mean = float(np.mean(np.abs(lap)))

    # We divide the image into 60x53 segments and calculate the
    # variance of the laplacian in each one.
    grid_height, grid_width = height // 8, width // 12

    patch_lap_vars = [
        np.var(lap[r * grid_height:(r + 1) * grid_height, 
                   c * grid_width:(c + 1) * grid_width])
        for r in range(8) for c in range(12)
    ]

    patch_lap_vars = np.array(patch_lap_vars, dtype=np.float32)

    patch_lap_median = float(np.median(patch_lap_vars))
    patch_lap_p25 = float(np.percentile(patch_lap_vars, 25))
    patch_lap_p90 = float(np.percentile(patch_lap_vars, 90))
    patch_lap_p90_to_med = float(patch_lap_p90 / (patch_lap_median + EPSILON))
    smooth_patches_lt2 = float(np.mean(patch_lap_vars < 2.0))
    smooth_patches_lt10 = float(np.mean(patch_lap_vars < 10.0))

    # We apply a Gaussian blur to our image. The idea is that blurring
    # an already blurred image changes little, while doing that to
    # a sharp image looses a lot of detail.
    img_blurred = cv2.GaussianBlur(img_gray, (0, 0), 2.0)

    # We calculate the Sobel gradients for the blurred and non-blurred image.
    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    blurred_grad_x = cv2.Sobel(img_blurred, cv2.CV_32F, 1, 0, ksize=3)
    blurred_grad_y = cv2.Sobel(img_blurred, cv2.CV_32F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    blurred_sobel_mag = np.sqrt(blurred_grad_x ** 2 + blurred_grad_y ** 2)

    grad_retention_global = np.mean(sobel_mag) / np.mean(blurred_sobel_mag) + EPSILON

    # Features for determining edge of splat.
    # We consider the border of the image, because this defect is almost
    # always most visible at the edges.
    border_h, border_w = int(0.11 * height), int(0.11 * width)

    border_top, border_bottom = img_gray[:border_h], img_gray[-border_h:]
    border_left, border_right = img_gray[:border_h], img_gray[-border_h:]

    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border_h] = True
    border_mask[-border_h:] = True
    border_mask[:border_w] = True
    border_mask[-border_w:] = True

    # If the edge of splat defect is present, the edges might be darker 
    # (or brighter in case of a white/bright bg) than the rest of the image.
    dark_pixels_ratio_global = float(np.mean(img_gray < 8.0))
    bright_pixels_ratio_global = float(np.mean(img_gray > 247.0))
    dark_pixels_ratio_border = float(np.mean(img_gray[border_mask]) < 8.0)
    bright_pixels_ratio_border = float(np.mean(img_gray[border_mask]) > 247.0)

    return np.array(
        [laplacian_var, laplacian_mean, patch_lap_median,
         patch_lap_p25, patch_lap_p90, patch_lap_p90_to_med,
         smooth_patches_lt2, smooth_patches_lt10,
         grad_retention_global],
         #dark_pixels_ratio_global, bright_pixels_ratio_global,
         #dark_pixels_ratio_border, bright_pixels_ratio_border], 
        dtype=np.float32)

def get_features(df):
    paths = []

    for img_set, img_name in df[['set', 'id']].itertuples(index=False):
        paths.append(os.path.join(IMG_DIR_PATH, img_set, img_name))

    # For each image, we calculate its features in a new process.
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(compute_image_features, paths, chunksize=32))

    return np.vstack(features)

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    x = get_features(df)
    y = df[['blur_fraction']].values.astype(np.float32)

    # We divide into a train set and a test set without splitting data
    # from the same set into different sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67)
    train_idx, test_idx = next(splitter.split(df, groups=df['set']))

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=67,
        n_jobs=-1
    )

    model.fit(x_train, y_train, verbose=50)

    y_pred = model.predict(x_test)

    y_test_bin = (y_test >= 0.5).astype(int)
    y_pred_bin = (y_pred >= 0.5).astype(int)

    f1 = f1_score(y_test_bin, y_pred_bin)

    print("F1: " + str(f1))