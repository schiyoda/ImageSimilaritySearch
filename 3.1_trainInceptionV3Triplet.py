import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import sys, os, time, gc, random, pickle
from sklearn.neighbors import NearestNeighbors
import requests, shutil
import tensorflow as tf
import keras

from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
import zipfile

training_data_dir = os.getenv("DATA_DIR")
training_results_dir = os.getenv("RESULT_DIR")

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

# training set triplet generator
def train_triplet_generator(df, batch_size=100, img_size=(224, 224), seed=42,
                            prefix='data/train/'):
    """ training set triplet generator
        it will generate 400 triplet images in total
    """
    # get images with only one training image landmark id and the rest landmark ids
    np.random.seed(seed)
    grouped = df[['landmark_id', 'image_id']].groupby('landmark_id').count().reset_index()
    unique_neg_ids = list(grouped[grouped['image_id'] == 1]['landmark_id'].values)
    rest_ids = list(grouped[grouped['image_id'] > 1]['landmark_id'].values)
    size = 400 * 2 - len(unique_neg_ids)
    zeros = np.zeros((batch_size, 3, 1), dtype=K.floatx())

    while True:
        # get positive and negative image landmark ids
        np.random.shuffle(rest_ids)
        candidate_ids = list(np.random.choice(rest_ids, size=size, replace=False))
        pos_landmark_ids = candidate_ids[:400]
        neg_landmark_ids = candidate_ids[400:] + unique_neg_ids
        np.random.shuffle(neg_landmark_ids)

        # transform landmark id into image id
        anc_img_ids = []
        pos_img_ids = []
        neg_img_ids = []

        for i in range(len(pos_landmark_ids)):
            tmp_pos_ids = df[df['landmark_id'] == pos_landmark_ids[i]]['image_id'].values
            anc_img_ids.append(tmp_pos_ids[0])
            pos_img_ids.append(tmp_pos_ids[1])

            tmp_neg_ids = df[df['landmark_id'] == neg_landmark_ids[i]]['image_id'].values
            neg_img_ids.append(tmp_neg_ids[0])

        # iterator to read batch images
        for j in range(len(pos_img_ids) // batch_size):
            batch_anc_img_ids = anc_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_pos_img_ids = pos_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_neg_img_ids = neg_img_ids[j * batch_size: (j + 1) * batch_size]

            # get images
            anc_imgs = []
            pos_imgs = []
            neg_imgs = []

            # iteratively read images
            for k in range(batch_size):
                anc_path = prefix + str(batch_anc_img_ids[k]) + '.jpg'
                pos_path = prefix + str(batch_pos_img_ids[k]) + '.jpg'
                neg_path = prefix + str(batch_neg_img_ids[k]) + '.jpg'

                tmp_anc_img = load_img(anc_path, target_size=img_size)
                tmp_anc_img = img_to_array(tmp_anc_img)
                anc_imgs.append(tmp_anc_img)

                tmp_pos_img = load_img(pos_path, target_size=img_size)
                tmp_pos_img = img_to_array(tmp_pos_img)
                pos_imgs.append(tmp_pos_img)

                tmp_neg_img = load_img(neg_path, target_size=img_size)
                tmp_neg_img = img_to_array(tmp_neg_img)
                neg_imgs.append(tmp_neg_img)

            # transform list to array
            anc_imgs = np.array(anc_imgs, dtype=K.floatx()) / 255.0
            pos_imgs = np.array(pos_imgs, dtype=K.floatx()) / 255.0
            neg_imgs = np.array(neg_imgs, dtype=K.floatx()) / 255.0

            yield [anc_imgs, pos_imgs, neg_imgs], zeros

# validation set triplet generator
def val_triplet_generator(df, batch_size=128, img_size=(224, 224),
                          seed=42, prefix='data/validation'):
    """ validation set triplet collector """

     # get images with only one image landmark id and the rest landmark ids
    grouped = df[['landmark_id', 'image_id']].groupby('landmark_id').count().reset_index()
    unique_neg_ids = list(grouped[grouped['image_id'] == 1]['landmark_id'].values)
    rest_ids = list(grouped[grouped['image_id'] > 1]['landmark_id'].values)
    size = 72 * 2 - len(unique_neg_ids)
    zeros = np.zeros((batch_size, 3, 1), dtype=K.floatx())

    while True:
        # get positive and negative image landmark ids
        np.random.seed(seed)
        candidate_ids = list(np.random.choice(rest_ids, size=size, replace=False))
        pos_landmark_ids = candidate_ids[:72]
        neg_landmark_ids = candidate_ids[72:] + unique_neg_ids
        np.random.shuffle(neg_landmark_ids)

        # transform landmark id into image id
        anc_img_ids = []
        pos_img_ids = []
        neg_img_ids = []

        for i in range(len(pos_landmark_ids)):
            tmp_pos_ids = df[df['landmark_id'] == pos_landmark_ids[i]]['image_id'].values
            anc_img_ids.append(tmp_pos_ids[0])
            pos_img_ids.append(tmp_pos_ids[1])

            tmp_neg_ids = df[df['landmark_id'] == neg_landmark_ids[i]]['image_id'].values
            neg_img_ids.append(tmp_neg_ids[0])

        # iterator to read batch images
        for j in range(len(pos_img_ids) // batch_size):
            batch_anc_img_ids = anc_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_pos_img_ids = pos_img_ids[j * batch_size: (j + 1) * batch_size]
            batch_neg_img_ids = neg_img_ids[j * batch_size: (j + 1) * batch_size]

            # get images
            anc_imgs = []
            pos_imgs = []
            neg_imgs = []

            # iteratively read images
            for k in range(batch_size):
                anc_path = prefix + str(batch_anc_img_ids[k]) + '.jpg'
                pos_path = prefix + str(batch_pos_img_ids[k]) + '.jpg'
                neg_path = prefix + str(batch_neg_img_ids[k]) + '.jpg'

                tmp_anc_img = load_img(anc_path, target_size=img_size)
                tmp_anc_img = img_to_array(tmp_anc_img)
                anc_imgs.append(tmp_anc_img)

                tmp_pos_img = load_img(pos_path, target_size=img_size)
                tmp_pos_img = img_to_array(tmp_pos_img)
                pos_imgs.append(tmp_pos_img)

                tmp_neg_img = load_img(neg_path, target_size=img_size)
                tmp_neg_img = img_to_array(tmp_neg_img)
                neg_imgs.append(tmp_neg_img)

            # transform list to array
            anc_imgs = np.array(anc_imgs, dtype=K.floatx()) / 255.0
            pos_imgs = np.array(pos_imgs, dtype=K.floatx()) / 255.0
            neg_imgs = np.array(neg_imgs, dtype=K.floatx()) / 255.0

            yield [anc_imgs, pos_imgs, neg_imgs], zeros

# Define base network for triplet network
def base_net(input_shape=(224, 224, 3)):
    """ define triplet network """
    # load pre-trained InceptionV3 model
    inception = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')

    # frozen shallow layers
    inception.trainable = True

    set_trainable = False
    for layer in inception.layers:
        if layer.name == 'mixed9':
            set_trainable = True

        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # define sequential model
    model = Sequential(name='base_net')
    model.add(inception)
    model.add(Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm'))

    return model

# Define triplet network
def triplet_net(base_model, input_shape=(224, 224, 3)):
    """ function to define triplet networks """
    # define input: anchor, positive, negative
    anchor = Input(shape=input_shape, name='anchor_input')
    positive = Input(shape=input_shape, name='positive_input')
    negative = Input(shape=input_shape, name='negative_input')

    # extract vector represent using CNN based model
    anc_vec = base_model(anchor)
    pos_vec = base_model(positive)
    neg_vec = base_model(negative)

    # stack outputs
    stacks = Lambda(lambda x: K.stack(x, axis=1), name='output')([anc_vec, pos_vec, neg_vec])

    # define inputs and outputs
    inputs=[anchor, positive, negative]
    outputs = stacks

    # define the triplet model
    model = Model(inputs=inputs, outputs=outputs, name='triplet_net')

    return model

# Define triplet loss
def triplet_loss(y_true, y_pred):
    """ function to compute triplet loss
        margin is predefined coded, manually change if needed
    """
    # define triplet margin
    margin = K.constant(0.2)
    zero = K.constant(0.0)

    # get the prediction vector
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # compute distance
    pos_distance = K.sum(K.square(anchor - positive), axis=1)
    neg_distance = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    partial_loss = pos_distance - neg_distance + margin
    full_loss = K.sum(K.maximum(partial_loss, zero), axis=0)

    return full_loss

# For reproduciable purpose
seed = 42
K.clear_session()
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(seed)
random.seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(seed)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Define Parameters
img_size = (224, 224, 3)  # target image size

# triplet image generator
train_generator = train_triplet_generator(train_df, batch_size=100, img_size=img_size[:2],
                                          seed=42, prefix='data/train/')

val_generator = val_triplet_generator(val_df, batch_size=64, img_size=img_size[:2],
                                      seed=42, prefix='data/validation/')

# Define triplet network model
base_model = base_net(input_shape=img_size)
base_model.summary()

triplet_model = triplet_net(base_model=base_model, input_shape=img_size)
triplet_model.summary()

# define learning scheduler
def lr_schedule(epoch):
    """ Learning rate schedule """
    lr = 1e-3
    if epoch > 80:
        lr *= 2e-1
    elif epoch > 60:
        lr *= 4e-1
    elif epoch > 40:
        lr *= 6e-1
    elif epoch > 20:
        lr *= 8e-1
    print('Learning rate: ', lr)
    return lr

# define optimizer
opt = keras.optimizers.Adam(lr=lr_schedule(50))

# Create call backs
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [lr_reducer, lr_scheduler]

# compile the model
triplet_model.compile(optimizer=opt, loss=triplet_loss)

# fit the mode
history = triplet_model.fit_generator(train_generator, steps_per_epoch=74, epochs=50,
                                      validation_data=val_generator, validation_steps=48,
                                      verbose=2, callbacks=callbacks)

base_model.save(training_data_dir + '/inception-base-0.2-model.h5')
pickle.dump(history.history, open(training_data_dir + '/inception-triplet-0.2-history.p', 'wb'))
_ = gc.collect()

if not os.path.exists(training_results_dir + '/model'):
    os.mkdir(training_results_dir + '/model')
base_model.save(training_results_dir + '/model/inception-base-0.2-model.h5')
