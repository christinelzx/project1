from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import data_crop
import math
from keras import losses
import tensorflow as tf

from data_pre import load_train_data
from data_pre import load_test_data

from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 96
img_depths = 36

smooth = 1.

def dice_coef(y_true, y_pred):  # dice coefficient
    y_true_f = K.flatten(y_true)  # flatten: tf.reshape(x, [-1])
    y_pred_f = K.flatten(y_pred)

    """Flatten a tensor.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor, reshaped into 1-D
    """

    intersection = K.sum(y_true_f * y_pred_f)

    total_voxels = 36.0 * 256 * 256 + 38.0 * 266 * 266 + 40.0 * 286 * 286

    w = 0.5  # weight
    a = y_true_f * K.log(y_pred_f)

    b = - total_voxels - K.sum(-y_true_f) * K.log(K.sum(-y_pred_f * 1))

    print("line 48", y_true_f.shape)
    print("line 49", y_true_f[0])
    print("line 50", y_true_f[35])
    print("line51", y_pred[0, 18, 29, 29, 0])
    print("line 52", y_pred.shape)  # (?, ?, 60, 60, 1)
    print(y_true.shape)  # (?, ?, 60, 60, 1)
    # print (y_true[18,27,127])
    loss = (1 / total_voxels) * (-w * a + b)

    return loss


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_fcn():
    # inputs = Input((36, 256)) # 36 256-dimensional input
    # inputs1 = Input((img_depths, img_rows, img_cols, 1))
    # inputs2 = Input((img_depths, img_rows, img_cols, 1))
    # inputs3 = Input((img_depths, img_rows, img_cols, 1))

    inputs1 = Input((36, 60, 60, 1))
    inputs2 = Input((38, 70, 70, 1))
    inputs3 = Input((40, 90, 90, 1))
    # print (inputs1) # Tensor("input_1:0", shape=(?, 36, 256, 256, 4), dtype=float32)
    # print ("inputs.shape", inputs1.shape) #(?, 36, 256, 256, 4)

    inputs = [0, 1, 2]
    inputs[0] = inputs1
    inputs[1] = inputs2
    inputs[2] = inputs3

    # print ("inputs, ", inputs.shape)


    '''
    input

    pathway 1 (patch size: 36*60*60):
    conv1-1 4,(5, 7, 7)
    pool1-1, maxpooling
    conv1-2 8, (3,5,5)
    conv1-3 16,(3,3,3)
    deconv, 16
    conv1-4, 16, (3, 3, 3)


    pathway 2 (patch size: 38*70*70, no padding):
    conv 2-1 4,(3,5,5)
    conv 2-2 8,(1,3,3)
    conv 2-3 16, (1,3,3)
    conv 2-4 16, (1,3,3)

    pathway 3 (patch size: 40*90*90, no padding):
    conv 3-1 4, (5,7,7)
    pool 3-1 maxpooling
    conv 3-2 8, (1,5,5)
    conv 3-3 16, (1,5,5)
    conv 3-4 16, (1,3,3)
    conv 3-5 16, (1,3,3)
    deconv, 16
    conv 3-6 16, (3,3,3)

    concatenate (conv1-4, conv2-4, conv3-6) 36*60*60
    conv 2, (3,3,3)

    output prediction
    '''

    conv1_1 = Conv3D(4, (5, 7, 7), activation='relu', padding='same')(
        inputs1)  # (batch, depth, rows, cols, channels) (None, 36, 256, 256, 4)
    print("conv1_1, ", conv1_1.shape)
    pool1_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1_1)
    print("pool1_1, ", pool1_1.shape)
    conv1_2 = Conv3D(8, (3, 5, 5), activation='relu', padding='same')(pool1_1)
    print("conv1_2, ", conv1_2.shape)
    conv1_3 = Conv3D(16, (3, 3, 5), activation='relu', padding='same')(conv1_2)
    print("conv1_3, ", conv1_3.shape)
    deconv1_1 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv1_3)
    print("deconv1_1, ", deconv1_1.shape)
    conv1_4 = Conv3D(16, (3, 3, 5), activation='relu', padding='same')(deconv1_1)
    print("conv1_4, ", conv1_4.shape, "\n")

    conv2_1 = Conv3D(4, (3, 5, 5), activation='relu', padding='valid')(
        inputs2)  # (batch, depth, rows, cols, channels) (None, 36, 256, 256, 4)
    print("conv2_1, ", conv2_1.shape)
    conv2_2 = Conv3D(8, (1, 3, 3), activation='relu', padding='valid')(conv2_1)
    print("conv2_2, ", conv2_2.shape)

    conv2_3 = Conv3D(16, (1, 3, 3), activation='relu', padding='valid')(conv2_2)
    print("conv2_3, ", conv2_3.shape)

    conv2_4 = Conv3D(16, (1, 3, 3), activation='relu', padding='valid')(conv2_3)
    print("conv2_4, ", conv2_4.shape, "\n")

    conv3_1 = Conv3D(4, (5, 7, 7), activation='relu', padding='valid')(
        inputs3)  # (batch, depth, rows, cols, channels) (None, 36, 256, 256, 4)
    print("conv3_1, ", conv3_1.shape)
    pool3_1 = MaxPooling3D(pool_size=(2, 2, 2))(conv3_1)
    print("pool3_1, ", pool3_1.shape)
    conv3_2 = Conv3D(8, (1, 5, 5), activation='relu', padding='valid')(pool3_1)
    print("conv3_2, ", conv3_2.shape)
    conv3_3 = Conv3D(16, (1, 5, 5), activation='relu', padding='valid')(conv3_2)
    print("conv3_3, ", conv3_3.shape)
    conv3_4 = Conv3D(16, (1, 3, 3), activation='relu', padding='valid')(conv3_3)
    print("conv3_4, ", conv3_4.shape)
    conv3_5 = Conv3D(16, (1, 3, 3), activation='relu', padding='valid')(conv3_4)
    print("conv3_5, ", conv3_5.shape)
    deconv3_1 = Conv3DTranspose(16, (3, 3, 3), strides=(2, 2, 2), padding='same')(conv3_5)
    print("deconv3_1, ", deconv3_1.shape)
    conv3_6 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(deconv3_1)
    # print("conv3_6, ", conv3_6.shape, "\n")
    print("shape of 3 pathway: ", conv1_4.shape, conv2_4.shape, conv3_6.shape)
    # exit(0)

    merge = concatenate([conv1_4, conv2_4, conv3_6], axis=0)
    print("merge: ", merge.shape)

    # merge = concatenate([conv1_4, conv2_4, conv3_6])

    conv4 = Conv3D(2, (3, 3, 3), activation='relu', padding='same')(merge)
    print("conv4: ", conv4.shape)

    conv5 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv4)
    print("conv5: ", conv5.shape)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=[conv5])
    #
    # print ("line 92, ", model.__dict__)

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    model.compile(optimizer=Adam(lr=1), loss=losses.binary_crossentropy)

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_depths, img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_depths, img_cols, img_rows), preserve_range=True)
        # Resize image to match a certain size.
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train, imgs_label_train = load_train_data()

    data_num = len(imgs_train)
    print (data_num)

    imgs_input1 = np.zeros((data_num, 36, 256, 256, 1))
    imgs_input2 = np.zeros((data_num, 38, 266, 266, 1))
    imgs_input3 = np.zeros((data_num, 40, 286, 286, 1))

    label_input1 = np.zeros((data_num, 36, 256, 256, 1))
    label_input2 = np.zeros((data_num, 38, 266, 266, 1))
    label_input3 = np.zeros((data_num, 40, 286, 286, 1))

    imgs_input1[:, :, :, :, 0] = imgs_train
    imgs_input2[:, 1:37, 5:261, 5:261, 0] = imgs_train
    imgs_input3[:, 2:38, 10:266, 10:266, 0] = imgs_train

    label_input1[:, :, :, :, 0] = imgs_label_train
    label_input2[:, 1:37, 5:261, 5:261, 0] = imgs_label_train
    label_input3[:, 2:38, 10:266, 10:266, 0] = imgs_label_train

    print("line184", imgs_input2.shape, imgs_input3.shape)

    # imgs_train = preprocess(imgs_train)
    # imgs_label_train = preprocess(imgs_label_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    # imgs_train = imgs_train[ ..., np.newaxis]

    imgs_label_train = imgs_label_train.astype('float32')
    imgs_label_train /= 255.  # scale masks to [0, 1]

    # imgs_label_train = imgs_label_train[..., np.newaxis]


    imgs = [0, 1, 2]
    labels = [0, 1, 2]
    imgs_input = np.array((32, 36, 60, 60, 1))

    imgs[0], labels[0] = data_crop.crop_image_and_label(imgs_input1, label_input1, 36, 60, channel=1)
    print("imgs[0],labels[0] \n, 204", imgs_input1.shape, imgs[0].shape)
    imgs[1], labels[1] = data_crop.crop_image_and_label(imgs_input2, label_input2, 38, 70, channel=1)
    print("imgs[1],labels[1] \n", imgs_input2.shape, imgs[1].shape)
    imgs[2], labels[2] = data_crop.crop_image_and_label(imgs_input3, label_input3, 40, 90, channel=1)
    print("imgs[2],labels[2] \n", imgs_input3.shape, imgs[2].shape)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_fcn()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print("Saved in weight.h5")

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)

    print("imgs type: ", type(imgs))  # list

    # imgs=np.array(imgs)
    print("imgs[0] type: ", type(imgs[0]))  # numpy.ndarray

    print("imgs[0].shape, ", imgs[0].shape)  # (197, 36, 60, 60, 1)
    print("imgs[1].shape, ", imgs[1].shape)
    print("imgs[2].shape, ", imgs[2].shape)

    # imgs_p[i] = resize(imgs[i], (img_depths, img_cols, img_rows), preserve_range=True)
    label_input = np.asarray(imgs_label_train)
    print(type(label_input))
    print(imgs_label_train.shape)
    label_input = label_input[..., np.newaxis]
    print(imgs_label_train.shape)
    # label_input = np.resize(label_input, (19,108,60,60,1))

    print(labels[0].shape, labels[1].shape, labels[2].shape)

    label_input=np.concatenate((labels[0],labels[0],labels[0]),axis=1)
    # label_input = labels[0]
    # print(label_input.shape)

    model.fit([imgs[0], imgs[1], imgs[2]], label_input, batch_size=4, epochs=1, verbose=1, shuffle=True,
              # validation_split=0.2,
              callbacks=[model_checkpoint])


def test_data():
    # model = get_fcn()
    #
    # # plot_model(model, to_file='model.png')
    # # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #
    # imgs_train = load_train_data()[0]
    # mean = np.mean(imgs_train)  # mean for data centering
    # std = np.std(imgs_train)  # std for data normalization
    #
    # print('-' * 30)
    # print('Loading and preprocessing test data...')
    # print('-' * 30)
    #
    # imgs_test = load_test_data()
    # # imgs_test = preprocess(imgs_test)
    #
    # imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    # imgs_test /= std
    #
    # test_num = len(imgs_test)
    #
    # imgs_test1 = np.zeros((test_num, 36, 256, 256, 1))
    # imgs_test2 = np.zeros((test_num, 38, 266, 266, 1))
    # imgs_test3 = np.zeros((test_num, 40, 286, 286, 1))
    #
    # imgs_test1[:, :, :, :, 0] = imgs_test
    # imgs_test2[:, 1:37, 5:261, 5:261, 0] = imgs_test
    # imgs_test3[:, 2:38, 10:266, 10:266, 0] = imgs_test
    #
    # imgs=[0,1,2]
    #
    # imgs[0] = data_crop.crop_image_and_label(imgs_test1, [], 36, 60, channel=1)[0]
    # imgs[1] = data_crop.crop_image_and_label(imgs_test2, [], 38, 70, channel=1)[0]
    # imgs[2] = data_crop.crop_image_and_label(imgs_test3, [], 40, 90, channel=1)[0]
    #
    # print (imgs[0].shape, imgs[1].shape, imgs[2].shape)
    #
    # print('-' * 30)
    # print('Loading saved weights...')
    # print('-' * 30)
    # model.load_weights('weights.h5')
    #
    # print('-' * 30)
    # print('Predicting masks on test data...')
    # print('-' * 30)
    #
    # print ()
    #
    # imgs_mask_test = model.predict([imgs[0], imgs[1], imgs[2]], batch_size = 2, verbose=1)
    #
    # np.save('imgs_mask_test.npy', imgs_mask_test)
    # print("imgs_mask_test shape: ", imgs_mask_test.shape)
    #
    # print('-' * 30)
    # print('Saving predicted masks to files...')
    # print('-' * 30)
    # pred_dir = 'preds'
    # if not os.path.exists(pred_dir):
    #     os.mkdir(pred_dir)

    imgs_mask_test=np.load('imgs_mask_test.npy')

    i = 0
    # zip(imgs_mask_test, imgs_id_test)

    print (imgs_mask_test.shape)
    for image in imgs_mask_test:
        image = (image[: , : , :, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join('pred', str(i) + '_pred.png'), image)
        i += 1


if __name__ == '__main__':
    # with tf.device(device_name_or_function='/gpu:16'):
    #     train_and_predict()
        test_data()