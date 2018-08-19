


import matplotlib.pyplot as plt
import numpy as np
import os

from skimage.io import imread


# path='/Users/LiuZhuoxi/Documents/Study/master/CUHK/project_2/Warwick QU Dataset (Released 2016_07_08)'
path = "data"
image_depths = 36
image_rows=256
image_cols=256
#


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    # Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.
    imgs_label_train = np.load('imgs_label_train.npy')
    return imgs_train, imgs_label_train

def load_test_data():
    imgs_test_a = np.load('imgs_test_a.npy')
    imgs_test_b = np.load('imgs_test_b.npy')
    return imgs_test_a, imgs_test_b



def create_train_data():
    # train_data_path = path+"/TrainingData"
    images_folders = os.listdir(path)
    # print(type(image_folders))

    total_train_img = 85

    imgs = []
    imgs_label = []

    print('-'*30)
    print('Creating training images...')
    print('-'*30)


    for i in range(0,total_train_img):   # image_folder: folders' name in 'training data', eg.: A1, A2...

        img = convert_to_array("train_"+str(i+1)+'.bmp')
        imgs.append(img)

        label = convert_to_array("train_"+str(i+1)+'_anno.bmp')
        imgs_label.append(label)


    imgs = np.array(imgs)
    imgs_label = np.array(imgs_label)

    print("imgs.shape, imgs[0].shape",imgs.shape, imgs[0].shape)
    print("imgs_label.shape, imgs_label[0].shape", imgs_label.shape, imgs_label[0].shape)


    np.save('imgs_train.npy', imgs)
    np.save('imgs_label_train.npy', imgs_label)
    # print('Saving to .npy files done.')
    return imgs,imgs_label


def create_test_data():

    test_a_number = 60
    test_b_number = 20

    test_a = []
    test_b = []

    print('-'*30)
    print('Creating test images...')
    print('-'*30)

    for a in range(0,test_a_number):
        img = convert_to_array("testA_"+str(a+1)+'.bmp')

        test_a.append(img)

    for b in range(0, test_b_number):
        img = convert_to_array("testB_"+str(b+1)+'_anno.bmp')
        test_b.append(img)

    test_a = np.array(test_a)
    test_b = np.array(test_b)

    np.save('imgs_test_a.npy', test_a)
    np.save('imgs_test_b.npy', test_b)

    print("line 108",test_a.shape, test_b.shape)
    print("line 109",test_a[0].shape, test_b[0].shape)
    # print('Saving to .npy files done.')
    # return imgs,imgs_label


def convert_to_array( file):

    image_file = os.path.join(path, file)
    # print (image_file)

    img = imread(image_file)
    return img



# convert_to_array('testA_1.bmp')
# convert_to_array('testA_1_anno.bmp')
if __name__ == '__main__':

    create_train_data()
    create_test_data()

# test_img = np.load('imgs_test_a.npy')
# print(test_img[0].shape)
#
# plt.imshow(test_img[0])
#
# # plt.suptitle("wat")
# plt.show()




