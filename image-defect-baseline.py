import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from xgboost import XGBRegressor
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, mean_absolute_error

# Constants
CSV_PATH = 'data/splatdata.csv'
IMG_DIR_PATH = 'data/images/'
FINAL_MODEL_SAVE_DIR ='model'
FINAL_MODEL_SAVE_NAME = 'baseline.json'
DISPLAY_FEATURE_IMPORTANCES = False

# The width and height of the grid we will be dividing the image into (measured
# in patches).
GRID_WIDTH = 12
GRID_HEIGHT = 8

EPSILON = 1e-6

LABELS = ['edge_fraction', 'visgauss_fraction', 
          'blur_fraction', 'unreal_fraction']

FEATURE_NAMES = [
        'laplacian_var', 'laplacian_mean', 'laplacian_mean_norm', 
        'patch_lap_median', 'patch_lap_p25', 'patch_lap_p90', 'smooth_patches_lt2',
        'smooth_patches_lt10', 'grad_retention_global', 'grad_ret_p25', 
        'grad_ret_p90', 'border_min_4_lap_to_inner', 'border_p15_lap_to_inner',
        'abs_dead_border_ratio', 'border_smooth_lap_max', 
        'border_min4_mf_to_inner', 'st_coh_mean', 'st_coh_p95', 'hf_coh_mean', 
        'hf_coh_p90', 'mf_coh_p90', 'needle_p99_to_mean', 'line_to_disk_ratio', 
        'color_cast_ratio', 'chroma_p95', 'patch_std_cv', 'black_void'
    ]

def _make_line_kernels(length=15):
    kernels = []
    c = length // 2
    for angle_idx in range(4):  # 0, 45, 90, 135 degrees
        theta = angle_idx * np.pi / 4.0
        k = np.zeros((length, length), dtype=np.uint8)
        dx = int(round(c * np.cos(theta)))
        dy = int(round(c * np.sin(theta)))
        cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness=1)
        kernels.append(k)
    return kernels


LINE_KERNELS_15 = _make_line_kernels(15)
DISK_KERNEL_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def compute_image_features(img_path):
    img = cv2.imread(img_path)

    height, width, _ = np.shape(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gray_std = float(np.std(img_gray))

    # Features for determining blur.

    # We calculate the blurrines using the laplacian operator.
    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    lap_abs = np.abs(lap)

    laplacian_var = float(np.var(lap))
    laplacian_mean = float(np.mean(np.abs(lap)))
    laplacian_mean_norm = float(laplacian_mean / (gray_std + EPSILON))

    # We divide the image into 60x53 segments and calculate the
    # variance of the laplacian in each one.
    patch_height, patch_width = height // GRID_HEIGHT, width // GRID_WIDTH

    patch_lap_vars = [
        np.var(lap[r * patch_height:(r + 1) * patch_height, 
                   c * patch_width:(c + 1) * patch_width])
        for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH)
    ]

    patch_lap_vars = np.array(patch_lap_vars, dtype=np.float32)

    patch_lap_median = float(np.median(patch_lap_vars))
    patch_lap_p25 = float(np.percentile(patch_lap_vars, 25))
    patch_lap_p90 = float(np.percentile(patch_lap_vars, 90))
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

    grad_retention_global = float(np.mean(blurred_sobel_mag) / 
                                  (np.mean(sobel_mag) + EPSILON))

    # Used later for detecting edge of splat.
    img_blurred_fine = cv2.GaussianBlur(img_gray, (0, 0), 1.5)
    img_blurred_wide = cv2.GaussianBlur(img_gray, (0, 0), 8.0)
    mf_signed = img_blurred_fine - img_blurred_wide
    mf_map = np.abs(mf_signed)

    # We divide the image into segments and calculate various properties for
    # each segment.
    patch_grad_retention = []
    grid_lap_var = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    grid_lap_mean = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    grid_mf_mean = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            r_0, r_1 = row * patch_height, (row + 1) * patch_height
            c_0, c_1 = column * patch_width, (column + 1) * patch_width

            patch_smag = sobel_mag[r_0:r_1, c_0:c_1]
            patch_bsmag = blurred_sobel_mag[r_0:r_1, c_0:c_1]
            patch_grad_retention.append(float(np.mean(patch_bsmag) / 
                                              (np.mean(patch_smag) + EPSILON)))

            grid_lap_mean[row, column] = np.mean(lap[r_0:r_1, c_0:c_1])

            grid_mf_mean[row, column] = np.mean(mf_map[r_0:r_1, c_0:c_1])

    patch_grad_retention = np.array(patch_grad_retention)

    grad_ret_p25 = np.percentile(patch_grad_retention, 25)
    grad_ret_p90 = np.percentile(patch_grad_retention, 90)

    # Features for determining edge of splat.
    # We consider the border of the image, because this defect is almost
    # always most visible at the edges.
    border_ring_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
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

    abs_dead_border_ratio = float(np.mean(border_lap_var < 2.0))

    # Now, rather than consider the border of our 8x12 grid, we consider a
    # border of the size (0.11 * image height) x (0.11 * image_width)
    border_h, border_w = int(0.11 * height), int(0.11 * width)
    border_laps = [lap_abs[:border_h, :], lap_abs[-border_h:, :],
                   lap_abs[:, :border_w], lap_abs[:, -border_w:]]

    # Maximum proportion of smooth pixels in the four borders.
    border_smooth_lap_max = float(np.max([np.mean(b < 1.0) for b in border_laps]))

    # Features for detecting visible Gaussians.
    hf_signed = img_gray - img_blurred
    hf_gx = cv2.Sobel(hf_signed, cv2.CV_32F, 1, 0, ksize=3)
    hf_gy = cv2.Sobel(hf_signed, cv2.CV_32F, 0, 1, ksize=3)
    hf_xx = cv2.GaussianBlur(hf_gx * hf_gx, (0, 0), 1.8)
    hf_yy = cv2.GaussianBlur(hf_gy * hf_gy, (0, 0), 1.8)
    hf_xy = cv2.GaussianBlur(hf_gx * hf_gy, (0, 0), 1.8)
    hf_coh = np.sqrt((hf_xx - hf_yy) ** 2 + 4.0 * (hf_xy ** 2)) / (hf_xx + hf_yy + 1.0)
    hf_coh_mean = float(np.mean(hf_coh))
    hf_coh_p90 = float(np.percentile(hf_coh, 90))

    mf_gx = cv2.Sobel(mf_signed, cv2.CV_32F, 1, 0, ksize=3)
    mf_gy = cv2.Sobel(mf_signed, cv2.CV_32F, 0, 1, ksize=3)
    mf_xx = cv2.GaussianBlur(mf_gx * mf_gx, (0, 0), 5.0)
    mf_yy = cv2.GaussianBlur(mf_gy * mf_gy, (0, 0), 5.0)
    mf_xy = cv2.GaussianBlur(mf_gx * mf_gy, (0, 0), 5.0)
    mf_coh = np.sqrt((mf_xx - mf_yy) ** 2 + 4.0 * (mf_xy ** 2)) / (mf_xx + mf_yy + 0.5)
    mf_coh_p90 = float(np.percentile(mf_coh, 90))

    open_disk = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, DISK_KERNEL_9)
    close_disk = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, DISK_KERNEL_9)
    tophat = img_gray - open_disk
    blackhat = close_disk - img_gray
    needle_resp = np.maximum(tophat, blackhat)
    needle_p99_to_mean = float(np.percentile(needle_resp, 99) / (np.mean(needle_resp) + EPSILON))

    sup_open_line = np.max([cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, k) 
                            for k in LINE_KERNELS_15], axis=0)
    
    inf_close_line = np.min([cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, k) 
                             for k in LINE_KERNELS_15], axis=0)
    
    line_only = np.maximum(np.maximum(0.0, sup_open_line - open_disk),
                           np.maximum(0.0, close_disk - inf_close_line))
    
    line_to_disk_ratio = float(np.mean(line_only) / (np.mean(needle_resp) + EPSILON))
    
    Jxx = cv2.GaussianBlur(grad_x * grad_x, (15, 15), 3.0)
    Jyy = cv2.GaussianBlur(grad_y * grad_y, (15, 15), 3.0)
    Jxy = cv2.GaussianBlur(grad_x * grad_y, (15, 15), 3.0)
    coherence = np.sqrt((Jxx - Jyy)**2 + 4.0 * (Jxy**2)) / (Jxx + Jyy + 1.0)

    st_coh_mean = float(np.mean(coherence))
    st_coh_p95 = float(np.percentile(coherence, 95))

    # Features for detecting the unreal image defect.

    # Black void - determines whether the image is largely made up of black
    # without any clear edges.
    black_void = float(np.mean((img_gray < 10.0) & (sobel_mag < 2.5)))

    # Determines whether a single color channel dominates.
    b_mean = float(np.mean(img[:, :, 0]))
    g_mean = float(np.mean(img[:, :, 1]))
    r_mean = float(np.mean(img[:, :, 2]))
    max_rgb = max(b_mean, g_mean, r_mean)
    min_rgb = min(b_mean, g_mean, r_mean)
    color_cast_ratio = float((max_rgb - min_rgb) / (max_rgb + min_rgb + 1.0))

    # We convert the image to the CIE LAB color space to look for chromatic
    # anomalies.
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_centered = img_lab[:, :, 1] - 128.0
    b_centered = img_lab[:, :, 2] - 128.0
    chroma = np.sqrt(a_centered ** 2 + b_centered ** 2)
    chroma_p95 = float(np.percentile(chroma, 95))

    # We measure the spatial disporportion of our image.
    h_crop, w_crop = GRID_HEIGHT * patch_height, GRID_WIDTH * patch_width
    gray_blocks = img_gray[:h_crop, :w_crop].reshape(GRID_HEIGHT, patch_height, 
                                                     GRID_WIDTH, patch_width)
    grid_gray_std = gray_blocks.std(axis=(1, 3)).astype(np.float32)
    patch_std_cv = float(np.std(grid_gray_std) / (np.mean(grid_gray_std) + 
                                                  EPSILON))

    return np.array(
         [laplacian_var, laplacian_mean, laplacian_mean_norm, 
         patch_lap_median, patch_lap_p25, patch_lap_p90, 
         smooth_patches_lt2, smooth_patches_lt10,
         grad_retention_global, grad_ret_p25, grad_ret_p90,
         border_min_4_lap_to_inner, border_p15_lap_to_inner,
         abs_dead_border_ratio, border_smooth_lap_max, border_min4_mf_to_inner,
         st_coh_mean, st_coh_p95, hf_coh_mean, hf_coh_p90, mf_coh_p90,
         needle_p99_to_mean, line_to_disk_ratio, color_cast_ratio, chroma_p95, 
         patch_std_cv, black_void],
        dtype=np.float32
    )

def get_features(df):
    paths = []

    for img_set, img_name in df[['set', 'id']].itertuples(index=False):
        paths.append(os.path.join(IMG_DIR_PATH, img_set, img_name))

    # For each image, we calculate its features in a new process.
    with ProcessPoolExecutor() as executor:
        features = list(tqdm(executor.map(compute_image_features, paths,
                                           chunksize=32), total=len(paths)))

    return np.vstack(features)

def log_loss(y_test, y_pred):
    # We clip y_pred to avoid getting log(0).
    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)

    return -np.mean(y_test * np.log(y_pred_clipped) + (1 - y_test) * 
                    np.log(1 - y_pred_clipped))

def find_best_f1_thresholds(y_true_bin, y_pred):
    best_thresholds = np.full(y_true_bin.shape[1], 0.5, dtype=np.float32)
    candidates_thresholds = np.linspace(0.2, 0.7, 50)

    for i in range(y_true_bin.shape[1]):
        best_score = -1.0
        for threshold in candidates_thresholds:
            score = f1_score(y_true_bin[:, i], (y_pred[:, i] >= threshold).astype(int),
                             zero_division=0)
            if score > best_score:
                best_score = score
                best_thresholds[i] = threshold

    return best_thresholds

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    print('Calculating image features')
    x = get_features(df)
    y = df[LABELS].values.astype(np.float32)

    groups = df['set'].values

    splitter = GroupKFold(n_splits=5)

    f1_scores = []
    bce_scores = []

    per_label_f1_scores = np.zeros((splitter.get_n_splits(), len(LABELS)), dtype=np.float32)
    per_label_mae = np.zeros((splitter.get_n_splits(), len(LABELS)), dtype=np.float32)

    feature_importances = np.zeros(len(FEATURE_NAMES), dtype=np.float32);

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups=groups)):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        model = XGBRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=10,
            reg_lambda=5.0,
            reg_alpha=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=67,
            n_jobs=-1,
        )

        model.fit(x_train, y_train, verbose=50)

        y_pred = model.predict(x_test)
        y_pred_train = model.predict(x_train)

        feature_importances += model.feature_importances_ / splitter.get_n_splits()

        y_train_bin = (y_train >= 0.5).astype(int)
        y_test_bin = (y_test >= 0.5).astype(int)

        bce = log_loss(y_test, y_pred)

        thresholds = find_best_f1_thresholds(y_train_bin, y_pred_train)
        y_pred_bin = (y_pred >= thresholds).astype(int)

        f1 = f1_score(y_test_bin, y_pred_bin, average='macro')
        f1_scores.append(f1)
        bce_scores.append(bce)

        print(f'\nFOLD {fold+1}/{splitter.get_n_splits()}')
        print(f'F1: {f1:.4f} | BCE: {bce:.4f}')

        for i, label in enumerate(LABELS):
            per_label_f1_scores[fold, i] = f1_score(y_test_bin[:, i], 
                                                    y_pred_bin[:, i], 
                                                    zero_division=0)
            per_label_mae[fold, i] = mean_absolute_error(y_test[:, i], 
                                                         y_pred[:, i])
            print(f'{label} | F1: {per_label_f1_scores[fold, i]:.4f} | '
                  f'MAE: {per_label_mae[fold, i]:.4f}')
            
    print('\nSummary')

    print(f'Mean F1: {np.mean(f1_scores):.4f} (+- {np.std(f1_scores):.4f})')
    print(f'Mean BCE: {np.mean(bce_scores):.4f} (+- {np.std(bce_scores):.4f})')

    print('\nPer-label metrics')

    for i, label in enumerate(LABELS):
        f1_mean = per_label_f1_scores[:, i].mean()
        mae_mean = per_label_mae[:, i].mean()
        print(f'{label:<20} | F1: {f1_mean:.4f} | MAE: {mae_mean:.4f}')

    importance_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    if DISPLAY_FEATURE_IMPORTANCES:
        print('\nFeature importances')
        print(importance_df)

    # We create a final model and save it.
    final_model = XGBRegressor(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=10,
        reg_lambda=5.0,
        reg_alpha=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=67,
        n_jobs=-1,
    )

    print ('\nTraining final model.')
    final_model.fit(x, y)
    print('Saving final model')

    if not os.path.isdir(FINAL_MODEL_SAVE_DIR):
        os.makedirs(FINAL_MODEL_SAVE_DIR)

    final_model.save_model(os.path.join(FINAL_MODEL_SAVE_DIR, FINAL_MODEL_SAVE_NAME))