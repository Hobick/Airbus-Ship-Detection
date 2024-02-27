import numpy as np

import cv2
import pandas as pd

# Local pathes
result_data_path = 'data/results/'

def mask_encode(image):
    # Transforming image into a line
    image = image.T.flatten()
    image = np.concatenate([[0], image, [0]])
    # Breacking line on blocks
    blocks = np.where(image[1:] != image[:-1])[0] + 1
    blocks[1::2] -= blocks[::2]
    # Concatinating blocks
    res = ' '.join(str(i) for i in blocks)
    return res

def create_sumbission():
    # Reading csv with test images
    df = pd.read_csv('sample_submission_v2.csv')

    index = 0

    for img_id in df['ImageId']:

        # Loading image and remove noise
        img = cv2.imread(result_data_path + img_id, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype = np.float32)
        img[img > 100] = 255
        img[img <= 100] = 0
        

        # Encoding the image and updating 'EncodedPixels value'
        encoded_mask = mask_encode(img)
        df.loc[df['ImageId'] == img_id, 'EncodedPixels'] = encoded_mask
        
        # Tracking the progress
        if index % 100 == 0:
            print('Encoded {0} masks'.format(index))
        
        index +=1

    # Saving submission file
    df.to_csv('sample_submission_v2.csv', index=False)


if __name__ == '__main__':
    create_sumbission()
