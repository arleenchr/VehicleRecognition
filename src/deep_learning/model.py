import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.applications.efficientnet import preprocess_input
import pickle

from deep_learning.preprocessing import *

cnn_model_file_path = os.path.join(os.getcwd(), 'src', 'deep_learning', 'model', 'cnn_model.pkl')

# Create model structure
def create_model_structure(train_gen):
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys())) # to define number of classes in dense layer

    # Create pre-trained model (efficientnetb3)
    base_model = tf.keras.applications.efficientnet.EfficientNetB7(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
    base_model.trainable = False

    model = Sequential([
        base_model,
        BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
        Dense(128, kernel_regularizer= regularizers.l2(0.016), activity_regularizer= regularizers.l1(0.006),
                    bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
        Dropout(rate= 0.45, seed= 123),
        Dense(class_count, activation= 'softmax')
    ])

    model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

    return model

def load_model(data_dir):
    # Load Model
    filepaths, labels = generate_data_paths(data_dir)
    df = create_df(filepaths, labels)

    train_df, valid_df, test_df = split_dataset(df)
    train_gen, valid_gen, test_gen = generate_image_data(train_df, valid_df, test_df)

    model = create_model_structure(train_gen)
    model.load_weights('./src/deep_learning/my_model_weights.h5')
    
    # Save Model
    with open(cnn_model_file_path, 'wb') as f:
        pickle.dump(model, f)

def predict_class(image_path, class_labels):
    # Load Model
    with open(cnn_model_file_path, 'rb') as f:
        model = pickle.load(f)
    
    # Predict
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label