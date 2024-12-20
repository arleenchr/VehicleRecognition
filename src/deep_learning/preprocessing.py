import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_data_paths(data_dir):
    
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
            
    return filepaths, labels

def create_df(filepaths, labels):
    Fseries = pd.Series(filepaths, name= 'filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis= 1)
    return df

def split_dataset(df):
    train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123)
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.6, shuffle= True, random_state= 123)
    return train_df, valid_df, test_df

# Create image data generator
def generate_image_data(train_df, valid_df, test_df):
    batch_size = 16
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)

    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function= scalar,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            brightness_range=[0.4,0.6],
                            zoom_range=0.3,
                            horizontal_flip=True,
                            vertical_flip=True)

    ts_gen = ImageDataGenerator(preprocessing_function= scalar,
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            brightness_range=[0.4,0.6],
                            zoom_range=0.3,
                            horizontal_flip=True,
                            vertical_flip=True)

    train_gen = tr_gen.flow_from_dataframe(train_df, 
                                        x_col= 'filepaths', 
                                        y_col= 'labels', 
                                        target_size= img_size, 
                                        class_mode= 'categorical',
                                        color_mode= 'rgb', 
                                        shuffle= True, 
                                        batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, 
                                        x_col= 'filepaths', 
                                        y_col= 'labels', 
                                        target_size= img_size, 
                                        class_mode= 'categorical',
                                        color_mode= 'rgb', 
                                        shuffle= True, 
                                        batch_size= batch_size)

    test_gen = ts_gen.flow_from_dataframe(test_df, 
                                        x_col= 'filepaths', 
                                        y_col= 'labels', 
                                        target_size= img_size, 
                                        class_mode= 'categorical',
                                        color_mode= 'rgb', 
                                        shuffle= False, 
                                        batch_size= test_batch_size)
    
    return train_gen, valid_gen, test_gen