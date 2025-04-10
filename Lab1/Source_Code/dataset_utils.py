import os
import cv2
import numpy as np
from imutils import paths
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size):
    image_paths = list(paths.list_images(data_dir))
    data, labels = [], []

    for img_path in tqdm(image_paths, desc="🔍 Loading images"):
        label = img_path.split(os.path.sep)[-2]
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)
        labels.append(label)

    data = np.array(data) / 255.0
    labels = np.array(labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    return X_train, X_test, y_train, y_test, le
