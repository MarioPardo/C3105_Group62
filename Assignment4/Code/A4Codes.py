#COMP3105 A4
#Mario Pardo 101286566
#Dante Farinon-Spezzano-101231566

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Multiply, Add, Subtract, Dropout, Reshape
from tensorflow.keras.models import Model, Sequential
import numpy as np
import pandas as pd  #added for importing data when testing



def learn(X, y):

    input_data = Input(shape=(X.shape[1],))
    top_data = Lambda(lambda x: x[:, :784])(input_data)
    middle_data = Lambda(lambda x: x[:, 784:1568])(input_data)
    bottom_data = Lambda(lambda x: x[:, 1568:])(input_data)

    # Find top digit even/odd
    reshaped_top_data = Reshape((28, 28, 1))(top_data)
    conv1_top = Conv2D(16, (5, 5), activation='relu')(reshaped_top_data)
    maxpool1_top = MaxPooling2D((2, 2))(conv1_top)
    conv2_top = Conv2D(32, (3, 3), activation='relu')(maxpool1_top)
    maxpool2_top = MaxPooling2D((2, 2))(conv2_top)
    flatten_top = Flatten()(maxpool2_top)
    dense1_top = Dense(256, activation='relu')(flatten_top)
    dropout1_top = Dropout(0.2)(dense1_top)
    dense2_top = Dense(10, activation='relu')(dropout1_top)
    dropout2_top = Dropout(0.2)(dense2_top)
    even_odd_output = Dense(1, activation='sigmoid')(dropout2_top)

    #Define model for reuse for middle and bottom
    mid_bottom_model = Sequential([
        Reshape((28, 28, 1)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(784, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])

    # Use the mid_bottom_model
    middle_output = mid_bottom_model(middle_data)
    bottom_output = mid_bottom_model(bottom_data)

    # Select which output to use
    ones_tensor = Lambda(lambda x: tf.ones_like(x))(even_odd_output)
    inverted_model1_output = Subtract()([ones_tensor, even_odd_output])
    middle_weighed = Multiply()([middle_output, even_odd_output])  
    bottom_weighed = Multiply()([bottom_output, inverted_model1_output]) 
    final_output = Add()([middle_weighed, bottom_weighed])

    model = Model(inputs=input_data, outputs=final_output)  # Single input

    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    def oneHot(num):
        one_hot_array = [0] * 10 
        if 0 <= num <= 9:
            one_hot_array[num] = 1
            return one_hot_array
        else:
            return one_hot_array
        
    # Preprocess data
    y = np.array([oneHot(yi) for yi in y])
    X = X / 255.0

    model.fit(X, y, epochs=15, batch_size=32, validation_split=0.2)
    return model

def classify(Xtest, model):
    Xtest = Xtest / 255.0
    y_pred = np.argmax(model.predict(Xtest), axis=1)
    return y_pred


