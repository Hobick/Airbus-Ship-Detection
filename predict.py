from keras.models import Model
from unet_model import unet
import numpy as np

import cv2
import pandas as pd

# Local pathes
test_data_path = 'data/test/'
result_data_path = 'data/results/'

def predict():
    
    # Loading the model
    model = unet()
    model.load_weights('model1.h5')

    # Reading csv with test images
    df = pd.read_csv('sample_submission_v2.csv')

    index = 0

    for img_id in df['ImageId']:

        # Loading image
        img = cv2.imread(test_data_path + img_id)

        # Preparing image for making predictions
        img = cv2.resize(img, (256, 256))              
        img = img.reshape(1, 256, 256, 3)
        img = np.array(img, dtype = np.float32)
        img /= 255

        # Predict
        pred = model(img).numpy() * 255

        # Preparing mask for saving
        pred = pred.reshape(256, 256, 1)
        pred = cv2.resize(pred, (768, 768))
        pred[pred >= 128] = 255
        pred[pred <= 128] = 0

        # Saving the mask
        cv2.imwrite(result_data_path + img_id, pred)

        # Tracking the progress
        if index % 100 == 0:
            print('Created {0} masks'.format(index))
        index +=1


if __name__ == '__main__':
    predict()