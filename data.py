import numpy as np
import cv2
import pandas as pd
import os

# Data pathes
masks_data_path = 'data/train/masks/'
images_data_path = 'data/train/images/'

# Images size
image_rows = 768
image_cols = 768


# Row-line-decoding function (Transformig string into 2D-mask)
def mask_decode(string):
    mask = np.zeros((image_rows, image_cols), dtype=np.byte)
    
    array = string.split(' ')
    for i in range(int(len(array) / 2)):
        x = int(array[2 * i])
        y = int(array[2 * i + 1])
        for j in range(y):    
            mask[min(x % image_cols + j, image_cols - 1)][x // image_cols] = 1
    return mask
    
# Main function for creating masks
def create_masks():
    
    #Reading csv with train images
    df = pd.read_csv('train_ship_segmentations_v2.csv')

    index = 0

    for img_id in df['ImageId'].unique():
        # Creating empty mask
        combined_mask = np.zeros((image_rows, image_cols), dtype=np.byte)

        contain_ship = False
        # Searching for all decoded masks and combining them 
        for mask in df[df['ImageId'] == img_id]['EncodedPixels']:
            if type(mask) == str:
                contain_ship = True
                combined_mask |= mask_decode(mask)
        
        # Creating and wirting masks into a file if there is a ship
        if contain_ship:
            cv2.imwrite(masks_data_path + img_id, combined_mask * 255)
        else:
            if os.path.exists(images_data_path + img_id):
                os.remove(images_data_path + img_id)        


        # Tracking process
        if index % 1000 == 0:
            print('Created {0} masks'.format(index))

        index +=1
        
    print('Complete')
        
if __name__ == '__main__':
    create_masks()
