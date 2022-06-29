"""
NOTES: Dataset details summary
- Image dataset, regression, 9 silos, 11k images in total
- Filename interpretation:
  For
    ADNI_016_S_0702_PT_AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20100927170934088_44_S92690_I193766.png
  the patient id is 0702, and image id is I193766.
- A silo is a manufacturer of a machine that took the images
- Each patient can have multiple images
- Each image is of size 160 x 160 by default (but scaled down below)
"""
import csv
import os
import sys
import json
import random
from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

MANUFACTURERS = {
    'CPS',
    'GE_MEDICAL_SYSTEMS',
    'GEMS',
    'HERMES',
    'MiE',
    'Multiple',
    'Philips',
    'Philips_Medical_Systems',
    'SIEMENS',
    'Siemens_ECAT',
    'Siemens_CTI'
}

DATA_DIR = './data/adni/'
DATA_PATH = DATA_DIR + 'adni_data2/'
LABEL_PATH = DATA_DIR + 'labels_2.txt'
ROOT_SEED = int((np.e ** np.euler_gamma) ** np.pi * 1000)


def read_image(path: str):
  with Image.open(path) as im:
    img = np.array(im.resize((32, 32)), dtype=np.float32)
  return img / 255.0


def read_data(seed=ROOT_SEED, test_split=0.25, save_dir=None):
  print(f'(seed={seed}) Reading ADNI data from {DATA_PATH} and {LABEL_PATH}...')
  # A nested map dic[patient_id][image_id] = label
  uid_mid_to_label = {}

  with open(LABEL_PATH, 'r') as f:
    labels = f.readlines()
  for label in labels:
    tokens = label.strip().split(',')
    uid, mid, value = tokens
    if uid in uid_mid_to_label:
      uid_mid_to_label[uid][mid] = float(value)
    else:
      uid_mid_to_label[uid] = {mid: float(value)}

  print('Populated label mapping...')
  # Partitioned by manufacturers (silos)
  x_trains, y_trains, x_tests, y_tests = [], [], [], []

  client_id = 0
  for client in os.listdir(DATA_PATH):
    if client not in MANUFACTURERS:
      continue

    print(f'Processing client "{client}"...')
    # Images and labels for each silo
    xx, yy = [], []
    images = os.listdir(os.path.join(DATA_PATH, client))
    for img_path in images:
      if not img_path.endswith('png'):  # ignore DS_Store
        continue
      uid, mid = img_path.strip().split('_')[3], img_path[:-4].strip().split('_')[-1]
      if uid in uid_mid_to_label:
        xx.append(read_image(os.path.join(DATA_PATH, client, img_path)))
        yy.append(uid_mid_to_label[uid][mid])

    # Skip if no samples for a silo
    assert len(xx) == len(yy)
    if len(xx) == 0:
      print(f'No data from client "{client}"')
      continue

    features, labels = np.array(xx)[..., None], np.array(yy)
    print('(unsplit) features:', features.shape, 'labels:', labels.shape)

    # Seeds controls the shuffling + train/test split
    client_seed = seed + client_id
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_split, random_state=client_seed)

    x_trains.append(x_train)
    y_trains.append(y_train)
    x_tests.append(x_test)
    y_tests.append(y_test)
    client_id += 1

  # List[List[array]], List[List[float]]
  x_trains = np.array(x_trains, dtype=object)
  y_trains = np.array(y_trains, dtype=object)
  x_tests = np.array(x_tests, dtype=object)
  y_tests = np.array(y_tests, dtype=object)

  if save_dir is not None:
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    np.save(os.path.join(save_dir, f'train_images_seed{seed}'), x_trains)
    np.save(os.path.join(save_dir, f'train_labels_seed{seed}'), y_trains)
    np.save(os.path.join(save_dir, f'test_images_seed{seed}'), x_tests)
    np.save(os.path.join(save_dir, f'test_labels_seed{seed}'), y_tests)
    print(f'Saved preprocessed ADNI dataset to {save_dir}')

  return x_trains, y_trains, x_tests, y_tests


if __name__ == "__main__":
  read_data()
