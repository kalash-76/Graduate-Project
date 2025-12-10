"""
Kaleb Ashmore
CSC 699 Masters Project
Dixie Alley TorNet CNN Model

Based on previous work by MIT's TorNet team and previous work by myself & a fellow student in CSC 606 Machine Learning.
Some code generated through ChatGPT's assistance and guidance, as well as MIT's DataLoaders.ipynb.
"""


import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sys

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, roc_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import AdamW



def build_cnn(shape):

    input_shape = shape[1:]  # (azimuth, range, 1)

    model = models.Sequential([
        Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # binary classification
        ])

    return model

def filter_nc_by_coords(file_list, bbox):
    """
    Filter nc files by radar site coords for Dixie Alley bounding box.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    filtered_files = []
    for f in file_list:
        try:
            ds = xr.open_dataset(f, decode_times=False)  # faster, don't decode times if not needed
            site_lat = ds.attrs.get('site_lat', None)
            site_lon = ds.attrs.get('site_lon', None)
            ds.close()

            if site_lat is None or site_lon is None:
                continue  # skip files without site coordinates

            if (lat_min <= site_lat <= lat_max) and (lon_min <= site_lon <= lon_max):
                filtered_files.append(f)

        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    return filtered_files


def load_vel_single_sweep(file_path, sweep_index=0):
    """
    Load VEL for one sweep and convert category to binary labels.
    """
    ds = xr.open_dataset(file_path)
    
    # Input
    X = ds['VEL'].isel(sweep=sweep_index).values
    X = X[..., np.newaxis]  # add channel dimension
    
    # Convert NaNs to 0 and then add mask for model
    mask = ~np.isnan(X)
    mask = mask.astype(float)
    X_filled = np.nan_to_num(X, nan=0.0)
    X_multichannel = np.concatenate([X_filled, mask], axis=-1)
    
    # Target
    cat = ds.attrs.get('category')  # string array of categories
    y_label = category_to_label(cat)
    y = np.full(X.shape[0], y_label, dtype=np.uint8)
    ds.close()
    # print(X_multichannel)
    return X_multichannel, y

def category_to_label(category):
    """
    Converts array of categories ('NUL', 'WRN', 'TOR') to binary labels:
    0 for NUL/WRN, 1 for TOR
    """
    if category == 'TOR':
        return 1
    else:  # NUL or WRN
        return 0



# Set path to tornet dataset
# Assumes .tar.gz files are extracted.
# this directory should contain catalog.csv, train/ , test/
data_root = os.path.dirname(os.getcwd())

# Load catalog of all TORNET samples
catalog = pd.read_csv(os.path.join(data_root,'catalog.csv'),parse_dates=['start_time','end_time'])

# set catalog to get training data from certain years
years = [2018,2022]

# Geographic bounding box for Dixie Alley
bbox = [-92.604280, 30.940283, -83.507187, 36.401681]

subset_train = catalog[
    (catalog.start_time.dt.year.isin(years)) & \
    (catalog['type']=='train') 
]

subset_test = catalog[
    (catalog.start_time.dt.year.isin(years)) & \
    (catalog['type']=='test')
]

# Get file list from TorNet Files in parent dir
if not os.path.exists('train_file_list.txt'):
    train_file_list = [os.path.join(data_root,f) for f in subset_train.filename]
    print('Found',len(train_file_list),'training files')
    train_file_list_filtered = filter_nc_by_coords(train_file_list, bbox)
    print('Found',len(train_file_list_filtered),'filtered training files')
    
    # Save list to file for future quick reference
    with open('train_file_list.txt', 'w') as f:
        for i in train_file_list_filtered:
            f.write(f'{i}\n')
else:
    with open('train_file_list.txt', 'r') as f:
        train_file_list_filtered = [line.strip() for line in f]

if not os.path.exists('test_file_list.txt'):
    test_file_list = [os.path.join(data_root,f) for f in subset_test.filename]
    print('Found',len(test_file_list),'testing files')
    test_file_list_filtered = filter_nc_by_coords(test_file_list, bbox)
    print('Found',len(test_file_list_filtered),'filtered testing files')
    with open('test_file_list.txt', 'w') as f:
        for i in test_file_list_filtered:
            f.write(f'{i}\n')
else:
    with open('test_file_list.txt', 'r') as f:
        test_file_list_filtered = [line.strip() for line in f]

# Get X and y from files
X_list, y_list = [], []

for f in train_file_list_filtered:
    X_file, y_file = load_vel_single_sweep(f, sweep_index=0)
    X_list.append(X_file)
    y_list.append(y_file)

X_train = np.concatenate(X_list, axis=0)
y_train = np.concatenate(y_list, axis=0)

print("X shape:", X_train.shape)
print("y shape:", y_train.shape)


X_list, y_list = [], []

for f in test_file_list_filtered:
    X_file, y_file = load_vel_single_sweep(f, sweep_index=0)
    X_list.append(X_file)
    y_list.append(y_file)

X_test = np.concatenate(X_list, axis=0)
y_test = np.concatenate(y_list, axis=0)


# Build and train CNN
cnn = build_cnn(X_train.shape)
cnn.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# cw = compute_class_weight('balanced', classes = np.array([0,1]), y=y_train)
# class_weights = {0: cw[0], 1: cw[1]}

history = cnn.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=8,
    shuffle=True,
    validation_split=0.30
    )

# Evaluate CNN

# Loss Curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


y_prob = cnn.predict(X_test)
y_pred = (y_prob > 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# ROC & AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], 'k--')  # diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()