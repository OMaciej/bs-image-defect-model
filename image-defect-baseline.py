import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from sklearn.model_selection import GroupShuffleSplit

# Constants
CSV_PATH = 'data/partialdata.csv'
IMG_DIR_PATH = 'data/images/'
LABELS = ['edge_fraction', 'visgauss_fraction', 'blur_fraction', 'unreal_fraction']

def compute_image_features(img_path):
    img = cv2.imread(img_path)

    # Features for determining blur.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    laplacian_var = float(np.var(lap))
    laplacian_mean = float(np.mean(np.abs(lap)))

    return np.array([laplacian_var, laplacian_mean], dtype=np.float32)

def get_features(df):
    paths = []

    for img_set, img_name in df[['set', 'id']].itertuples(index=False):
        paths.append(os.path.join(IMG_DIR_PATH, img_set, img_name))

    # For each image, we calculate its features in a new process.
    with ProcessPoolExecutor() as executor:
        features = list(executor.map(compute_image_features, paths, chunksize=32))

    return np.vstack(features)

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH, nrows=128)

    x = get_features(df)
    y = df[LABELS].values.astype(np.float32)

    # We divide into a train set and a test set without splitting data
    # from the same set into different sets.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=67)
    test_idx, train_idx = next(splitter.split(df, groups=df['set']))

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(x_train)
    print()
    print(x_test)