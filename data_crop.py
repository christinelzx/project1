import numpy as np
import copy
import data_pre


def crop_image_and_label( image_data_list,label_data_list, input_depth, input_size, channel=1):


        # image_data_list, label_data_list = data_pre.create_train_data()

        origin_image_size = len(image_data_list[0][0]) # 256
        origin_image_depth = len(image_data_list[0]) # 36

        depth_padding_num= int((input_depth-origin_image_depth)/2) # 36 → 0, 38,1; 40, 2
        depth_padding = np.zeros((origin_image_size, origin_image_size))

        # print(depth_padding_num, depth_padding.shape)

        print ("image_data_list.shape, ", image_data_list.shape)
        print("image_data_list[0].shape, ", image_data_list[0].shape)

        # add padding in depth
        # for i in range(0,int(depth_padding_num)+1):
        #     image_data_list = np.insert(image_data_list, 0, depth_padding, axis=1)
        #     image_data_list = np.insert(image_data_list, len(image_data_list[0]), depth_padding, axis=1)
        #
        #     label_data_list = np.insert(label_data_list, 0, depth_padding, axis=1)
        #     label_data_list = np.insert(label_data_list, len(label_data_list[0]), depth_padding, axis=1)


        max_patch = 197      # 256-60+1
        padding_num = int((max_patch-1+input_size - origin_image_size)/2)

        # padding = np.zeros((36,256,256))
        # print(padding.shape)

        # print (padding.ndim, image_data_list[0].ndim, image_data_list[0][0].ndim)

        # add padding in rows and columns
        # for m in range(0,input_depth+1):
        #     for n in range(0, padding_num+1):
        #         image_data_list = np.concatenate((padding, image_data_list[0]))



        print("image_data_list.shape", image_data_list.shape)
        # print("label_data_list.shape", label_data_list.shape)

        # TODO
        batch_size = 197
        # batch_size =19
        print (batch_size)

        # image_patch = np.zeros([batch_size, input_depth, input_size, input_size, channel]).astype('float32')
        # print ("line26", image_patch)
        # label_patch = np.zeros([batch_size, input_depth, input_size, input_size]).astype('int32')

        image_patch = np.zeros([batch_size, input_depth, input_size, input_size]).astype('float32')
        label_patch = np.zeros([batch_size, input_depth, input_size, input_size]).astype('int32')

        for a in range(batch_size):
        # for i in range(batch_size):

          for b in range(batch_size):
            i=0
            m=0

            # image cropped
            crop_image = image_data_list[i].astype('float32')

            # cropping
            crop_start = np.array([a,b,b])
            image_temp = copy.deepcopy(
                crop_image[
                    crop_start[0]:crop_start[0] + input_depth,
                    crop_start[1]:crop_start[1] + input_size,
                    crop_start[2]:crop_start[2] + input_size
                ]
            )
            image_patch= image_patch[..., np.newaxis]
            image_patch[i] = image_temp

            # print ("image_temp, ", image_temp)
            # print ("crop_image, ", crop_image)
            print ("line 83, image_temp.shape, ", image_temp.shape)
            if (len(label_data_list)!=0) :
                crop_label = label_data_list[i].astype('int32')

                label_temp = copy.deepcopy(
                    crop_label[
                        crop_start[0]:crop_start[0] + input_depth,
                        crop_start[1]:crop_start[1] + input_size,
                        crop_start[2]:crop_start[2] + input_size
                    ]
                )
                print ("line 91, image temp shape", image_temp.shape)
                print ("line 92, patch shape", image_patch.shape)
                # exit(0)
                #
                label_patch= label_patch[..., np.newaxis]
                # print(image_patch[i,:,:,:,:])

                #
                # print("image patch shape, ", image_patch.shape)
                # print("label_patch shape, ", label_patch.shape)
                # print (image_patch)

                label_patch[i] = label_temp




            return np.array(image_patch), np.array(label_patch)

# if __name__ == '__main__':
    # crop_image_and_label( 0, 0, 38, 70, channel=1)

