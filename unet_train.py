from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
# import tensorflow as tf
from keras import losses
import nibabel as nib


from data_pre import load_train_data, load_test_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
#
# img_rows = 96
# img_cols = 96
# img_depths = 36

img_rows = 96
img_cols = 96
img_depths = 16

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_depths, img_rows, img_cols, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1), loss=dice_coef, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_depths, img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_depths, img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    # with tf.device(device_name_or_function='gpu'):
    # with tf.device('/gpu:16'):
        print('-'*30)
        print('Loading and preprocessing train data...')
        print('-'*30)
        imgs_train, imgs_mask_train = load_train_data()
    #
        imgs_train = preprocess(imgs_train)
        imgs_mask_train = preprocess(imgs_mask_train)

        imgs_train = imgs_train.astype('float32')
        mean = np.mean(imgs_train)  # mean for data centering
        std = np.std(imgs_train)  # std for data normalization
    #
        imgs_train -= mean
        imgs_train /= std
    #
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_mask_train /= 255.  # scale masks to [0, 1]

        print('-'*30)
        print('Creating and compiling model...')
        print('-'*30)
        model = get_unet()
        model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
        # model_checkpoint = ModelCheckpoint('weights1.h5', monitor='val_loss', save_best_only=True)


        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=1, verbose=1, shuffle=True,
                  # validation_split=0.2,
                  callbacks=[model_checkpoint])

        model.save("weights1.h5")

        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        imgs_test = load_test_data()
        imgs_test = preprocess(imgs_test)
    #
        imgs_test = imgs_test.astype('float32')
        imgs_test -= mean
        imgs_test /= std

        print('-'*30)
        print('Loading saved weights...')
        print('-'*30)
        model.load_weights('weights1.h5')

        print('-'*30)
        print('Predicting masks on test data...')
        print('-'*30)
        imgs_mask_test = model.predict(imgs_test, verbose=1)
        # np.save('imgs_mask_test1.npy', imgs_mask_test)
        nib.save('imgs_mask_test1.nii', imgs_mask_test)

        print('-' * 30)
        print('Saving predicted masks to files...')
        print('-' * 30)
        pred_dir = 'preds'
        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)

        i=0
        for image in imgs_mask_test:
            # image = (image[:, :, 0] * 255.).astype(np.uint8)
            image = (image[:,:, :, 0] * 255.).astype(np.float32)

            imsave(os.path.join(pred_dir, str(i) + '_pred1.png'), image)
            i+=1

if __name__ == '__main__':
    # import tensorflow as tf

    # with tf.device(device_name_or_function='gpu'):
    # with tf.device('/gpu:16'):
        train_and_predict()