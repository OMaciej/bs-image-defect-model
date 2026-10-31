import os
import cv2
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

# Constants
CSV_PATH = 'data/splatdata.csv'
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

    grad_retention_global = float(np.mean(blurred_sobel_mag) / (np.mean(sobel_mag) + EPSILON))

    # Used later for detecting edge of splat.
    img_blurred_fine = cv2.GaussianBlur(img_gray, (0, 0), 1.5)
    img_blurred_wide = cv2.GaussianBlur(img_gray, (0, 0), 8.0)
    mf_map = np.abs(img_blurred_fine - img_blurred_wide)

    # We divide the image into segments and calculate various properties for
    # each segment.
    patch_grad_retention = []
    grid_lap_var = np.zeros((8, 12), dtype=np.float32)
    grid_lap_mean = np.zeros((8, 12), dtype=np.float32)
    grid_mf_mean = np.zeros((8, 12), dtype=np.float32)

    for row in range(8):
        for column in range(12):
            r_0, r_1 = row * grid_height, (row + 1) * grid_height
            c_0, c_1 = column * grid_width, (column + 1) * grid_width

            patch_smag = sobel_mag[r_0:r_1, c_0:c_1]
            patch_bsmag = blurred_sobel_mag[r_0:r_1, c_0:c_1]
            patch_grad_retention.append(float(np.mean(patch_bsmag) / (np.mean(patch_smag) + EPSILON)))

            grid_lap_var[row, column] = np.var(lap[r_0:r_1, c_0:c_1])
            grid_lap_mean[row, column] = np.mean(lap[r_0:r_1, c_0:c_1])

            grid_mf_mean[row, column] = np.mean(mf_map[r_0:r_1, c_0:c_1])

    patch_grad_retention = np.array(patch_grad_retention)

    grad_ret_p25 = np.percentile(patch_grad_retention, 25)
    grad_ret_p90 = np.percentile(patch_grad_retention, 90)

    # Features for determining edge of splat.
    # We consider the border of the image, because this defect is almost
    # always most visible at the edges.
    border_ring_mask = np.zeros((8, 12), dtype=bool)
    border_ring_mask[0, :] = True
    border_ring_mask[-1, :] = True
    border_ring_mask[:, 0] = True
    border_ring_mask[:, -1] = True

    border_mf_map = grid_mf_mean[border_ring_mask]
    inner_mf_map = grid_mf_mean[~border_ring_mask]

    border_lap_var = grid_lap_var[border_ring_mask]
    inner_lap_var = grid_lap_var[~border_ring_mask]

    inner_lap_med = float(np.median(inner_lap_var))
    inner_mf_med = float(np.median(inner_mf_map))

    border_smoothest_4_lap_med = float(np.median(np.sort(border_lap_var)[:4]));
    border_smoothest_4_mf_med = float(np.median(np.sort(border_mf_map)[:4]))
    
    # Ratio of the four smoothest patches to the inner image's median.
    border_min_4_lap_to_inner = float(border_smoothest_4_lap_med / 
                                      (inner_lap_med + EPSILON))

    # Ratio of the 15th percentile smoothestspatches to the inner image's median.
    border_p15_lap_to_inner = float(np.percentile(border_lap_var, 15) / 
                                    (inner_lap_med + EPSILON))

    border_min4_mf_to_inner = float(border_smoothest_4_mf_med / (inner_mf_med + EPSILON))

    # Proportion of border patches that are at least 5 times smoother than the 
    # inner image.
    rel_dead_border_ratio = float(np.mean(border_lap_var < 0.2 * inner_lap_med))

    abs_dead_border_ratio = float(np.mean(border_lap_var < 0.2))

    # Now, rather than consider the border of our 8x12 grid, we consider a
    # border of the size (0.11 * image height) x (0.11 * image_width)
    border_h, border_w = int(0.11 * height), int(0.11 * width)
    border_laps = [lap_abs[:border_h, :], lap_abs[-border_h:, :],
                   lap_abs[:, :border_w], lap_abs[:, -border_w:]]

    # Maximum proportion of smooth pixels in the four borders.
    border_smooth_lap_max = float(np.max([np.mean(b < 1.0) for b in border_laps]))

    return np.array(
        #[laplacian_var, laplacian_mean, patch_lap_median,
        # patch_lap_p25, patch_lap_p90, patch_lap_p90_to_med,
        # smooth_patches_lt2, smooth_patches_lt10,
        # grad_retention_global, grad_ret_p25, grad_ret_p90,
        [border_min_4_lap_to_inner, border_p15_lap_to_inner,
         abs_dead_border_ratio, rel_dead_border_ratio,
         border_smooth_lap_max, border_min4_mf_to_inner],
        dtype=np.float32)

def get_features(df):
    paths = []

    for img_set, img_name in df[['set', 'id']].itertuples(index=False):
        paths.append(os.path.join(IMG_DIR_PATH, img_set, img_name))

    # For each image, we calculate its features in a new process.
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(compute_image_features, paths, chunksize=32))

    #features = []
    #for path in paths:
    #    features.append(compute_image_features(path))

    return np.vstack(features)

def log_loss(y_test, y_pred):
    # We clip y_pred to avoid getting log(0).
    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)

    return -np.mean(y_test * np.log(y_pred_clipped) + (1 - y_test) * np.log(1 - y_pred_clipped))

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    x = get_features(df)
    y = df['edge_fraction'].values.astype(np.float32)

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

    f1 = f1_score(y_test_bin, y_pred_bin, average='binary')
    bce = log_loss(y_test, y_pred)

    print('F1: ' + str(f1))
    print('BCE: ' + str(bce))

    importance_df = pd.DataFrame({
        #'feature': ['laplacian_var', 'laplacian_mean', 'patch_lap_median',
        # 'patch_lap_p25', 'patch_lap_p90', 'patch_lap_p90_to_med',
        # 'smooth_patches_lt2', 'smooth_patches_lt10',
        # 'grad_retention_global', 'grad_ret_p25', 'grad_ret_p90'],
        'feature': ['border_min_4_lap_to_inner', 'border_p15_lap_to_inner',
         'abs_dead_border_ratio', 'rel_dead_border_ratio', 'border_smooth_lap_max',
         'border_min4_mf_to_inner'],
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(importance_df)