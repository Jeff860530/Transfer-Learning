from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
 
gener=datagen.flow_from_directory('./train/',#类别子文件夹的上一级文件夹
                                         batch_size=197,
                                         shuffle=False,
                                         save_to_dir='./generator_result/',
                                         save_prefix='trans_',
                                         save_format='jpg')
for i in range(1):
    gener.next()
