import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

from skimage.io import imsave, imread


path = "DataIVD"
#
# file = "A1/A1_fat.nii"
# # file = "A1"
#
#
# def show_slices(slices):
#    fig, axes = plt.subplots(1, len(slices))
#
#    for i, slice in enumerate(slices):
#        axes[i].imshow(slice.T, cmap="gray", origin="lower")
#
#
# anat_img = nib.load(path+"/TrainingData/"+file)
# anat_img_data = anat_img.get_data()
# anat_img_data.shape # (36, 256, 256)
#
# # print("anat_img", anat_img)
# # print("anat_img_data", anat_img_data)
# print("[17,127,127]", anat_img_data[17,127,127]) #
#
# # print ("[anat_img_data[2, :, :], ", anat_img_data[2, :, :])
#
# show_slices([anat_img_data[17, :, :],
#              anat_img_data[:, 127, :],
#              anat_img_data[:, :, 127]])
# # plt.suptitle("wat")
# plt.show()
# if __name__ == '__main__':
#  show_slices()

# (test image with size 40*304*304)

image_depths = 36
image_rows=256
image_cols=256
#


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    # Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.
    imgs_label_train = np.load('imgs_label_train.npy')
    return imgs_train, imgs_label_train


def create_train_data():
    # train_data_path = path+"/TrainingData"
    images_folders = os.listdir(path+"/TrainingData")

    total = len(images_folders)
    types = ['fat', 'inn', 'opp', 'wat']

    imgs = np.ndarray((72, image_depths, image_rows, image_cols), dtype=np.uint8)
    imgs_label = np.ndarray((72, image_depths, image_rows, image_cols), dtype=np.uint8)

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    i = 0

    for image_folder in images_folders:   # image_folder: folders' name in 'training data', eg.: A1, A2...

        if image_folder.startswith('A'):
            # raw images
            for type in types:
                print (image_folder, type)
                img = convert_to_array(image_folder, "TrainingData",type)
                imgs[i] = img
    #
                label = convert_to_array(image_folder, "TrainingData",'Labels')
                imgs_label[i] = label
                print (i)
                i+=1

            # print('Loading done.', image_folder)

    # print ('i, j', i, j)
    print ("imgs shape: ", imgs.shape) #(18*4, 36, 256, 256)
    # print (imgs)
    print ("label shape: ", imgs_label.shape)
    # print(imgs[0,17,127,127])
    np.save('imgs_train.npy', imgs)
    np.save('imgs_label_train.npy', imgs_label)
    print('Saving to .npy files done.')
    return imgs,imgs_label

def create_test_data():
    # test_data_path = path + "/TestData"
    images_folders = os.listdir(path+"/TestData")

    print (images_folders)

    total = len(images_folders)
    types = ['fat', 'inn', 'opp', 'wat']

    test_imgs = np.ndarray((24, image_depths, image_rows, image_cols), dtype=np.uint8)
    # imgs_id = np.ndarray((total,))

    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    i=0
    for test_folder in images_folders:
        # image_folder: folders' name in 'training data', eg.: A1, A2...
        # print("line 131", i)
        if (test_folder.startswith('A')):
            for type in types:
                # raw images
                test_img = convert_to_array(test_folder, "TestData/" , type)
                test_imgs[i] = test_img
                print (i)
                i += 1

    # print ('i, j', i, j)
    print("test imgs shape: ", test_imgs.shape)  # (18*4, 36, 256, 256)
    # print (imgs)
    np.save('imgs_test.npy', test_imgs)
    # np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    # imgs_id = np.load('imgs_id_test.npy')
    # return imgs_test, imgs_id
    return imgs_test


def convert_to_array(folder,type, file):

    image_name = folder + '_' + file + '.nii'
    image_file = os.path.join(path, type,folder, image_name)

    anat_img = nib.load(image_file)
    anat_img_data = anat_img.get_data()
    img = anat_img_data
    img = np.array([img])
    # print(folder, file)
    # print (image_file, img.shape)
    return img


if __name__ == '__main__':
    create_train_data()
    create_test_data()









'''
3d nii save as 3d numpy array

import numpy as np
import nibabel as nib

img = nib.load(example_filename)

a = np.array(img.dataobj)
'''








'''
3d nii save as 3d numpy array

import numpy as np
import nibabel as nib

img = nib.load(example_filename)

a = np.array(img.dataobj)
'''