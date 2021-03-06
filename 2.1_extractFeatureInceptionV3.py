import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import sys, os, time, gc
from sklearn.neighbors import NearestNeighbors
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3
import zipfile

training_data_dir = os.getenv("DATA_DIR")
training_results_dir = os.getenv("RESULT_DIR")

if not os.path.exists(training_data_dir + '/data'):
    os.mkdir(training_data_dir + '/data')

with zipfile.ZipFile(training_data_dir + '/data.zip') as existing_zip:
    existing_zip.extractall('.')

train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/validation.csv')
test_df = pd.read_csv('data/test.csv')

print('Train:\t\t', train_df.shape)
print('Validation:\t', val_df.shape)
print('Test:\t\t', test_df.shape)

print('\nTrain Landmarks:\t', len(train_df['landmark_id'].unique()))
print('Validation Landmarks:\t', len(val_df['landmark_id'].unique()))
print('Test Landmarks:\t\t', len(test_df['landmark_id'].unique()))


# Load pre-trained InceptionV3
img_size = (224, 224, 3)
inception = InceptionV3(include_top=False, weights='imagenet', input_shape=img_size, pooling='avg')
inception.summary()

# Process training images
if not os.path.exists(training_data_dir + '/data/train_naive_inceptionV3_features.npy'):
    train_imgs = np.zeros(shape=(len(train_df), 2048), dtype=np.float32)
    img_ids = train_df['image_id'].values
    steps = 200
    for i in range(0, len(train_df), steps):
        tmp_imgs = []
        print('\nProcess: {:10d}'.format(i))

        start = i
        end = min(len(train_df), i + steps)
        for idx in range(start, end):
            if idx % 10 == 0:
                print('=', end='')

            img_id = img_ids[idx]
            path = 'data/train/' + str(img_id) + '.jpg'
            img = load_img(path, target_size=img_size[:2])
            img = img_to_array(img)
            tmp_imgs.append(img)

        tmp_imgs = np.array(tmp_imgs, dtype=np.float32) / 255.0
        tmp_prediction = inception.predict(tmp_imgs)
        train_imgs[start: end, ] = tmp_prediction
        _ = gc.collect()

    # Save to disk
    np.save(training_data_dir + '/data/train_naive_inceptionV3_features.npy', train_imgs)

# Process validation images
if not os.path.exists(training_data_dir + '/data/validation_naive_inceptionV3_features.npy'):
    val_imgs = np.zeros(shape=(len(val_df), 2048), dtype=np.float32)
    img_ids = val_df['image_id'].values
    steps = 200
    for i in range(0, len(val_df), steps):
        tmp_imgs = []
        print('\nProcess: {:10d}'.format(i))

        start = i
        end = min(len(val_df), i + steps)
        for idx in range(start, end):
            if idx % 10 == 0:
                print('=', end='')

            img_id = img_ids[idx]
            path = 'data/validation/' + str(img_id) + '.jpg'
            img = load_img(path, target_size=img_size[:2])
            img = img_to_array(img)
            tmp_imgs.append(img)

        tmp_imgs = np.array(tmp_imgs, dtype=np.float32) / 255.0
        tmp_prediction = inception.predict(tmp_imgs)
        val_imgs[start: end, ] = tmp_prediction
        _ = gc.collect()

    # Save to disk
    np.save(training_data_dir + '/data/validation_naive_inceptionV3_features.npy', val_imgs)

# Process test images
if not os.path.exists(training_data_dir + '/data/test_naive_inceptionV3_features.npy'):
    test_imgs = np.zeros(shape=(len(test_df), 2048), dtype=np.float32)
    img_ids = test_df['image_id'].values
    steps = 200
    for i in range(0, len(test_df), steps):
        tmp_imgs = []
        print('\nProcess: {:10d}'.format(i))

        start = i
        end = min(len(test_df), i + steps)
        for idx in range(start, end):
            if idx % 10 == 0:
                print('=', end='')

            img_id = img_ids[idx]
            path = 'data/test/' + str(img_id) + '.jpg'
            img = load_img(path, target_size=img_size[:2])
            img = img_to_array(img)
            tmp_imgs.append(img)

        tmp_imgs = np.array(tmp_imgs, dtype=np.float32) / 255.0
        tmp_prediction = inception.predict(tmp_imgs)
        test_imgs[start: end, ] = tmp_prediction
        _ = gc.collect()

    # Save to disk
    np.save(training_data_dir + '/data/test_naive_inceptionV3_features.npy', test_imgs)
