from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from unet_model import unet
from random import randint

# Local filepathes
train_path = 'data/train/'
checkpoint_filepath = 'checkpoints/'

def train():
    model = unet()


    #Loading weights either from model file or checkpoint file (should to uncomment the second line)
    model.load_weights('model1.h5')
    #model.load_weights(checkpoint_filepath)

    # Preprocessing data params for ImageDataGenerator
    datagen_args = dict(
        rescale=1./255.,
        validation_split=0.2
    )
    
    #Creating data-generators
    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)

    #
    args = dict(
        class_mode=None,
        shuffle = False,
        batch_size = 32,
        seed = randint(0, 1000)
    )

    #Creating directory-flowing generators for train/validation inputs(images)/outputs(masks)
    train_image_generator = image_datagen.flow_from_directory(train_path,
                                                        classes=['images'],                                                        
                                                        color_mode = 'rgb',
                                                        subset='training',
                                                        **args)
    
    train_mask_generator = mask_datagen.flow_from_directory(train_path,
                                                      classes=['masks'],
                                                      color_mode = 'grayscale',
                                                      subset='training',
                                                      **args)
    
    validation_image_generator = image_datagen.flow_from_directory(train_path,
                                                        classes=['images'],                                                        
                                                        color_mode = 'rgb',
                                                        subset='validation',
                                                        **args)
    
    validation_mask_generator = mask_datagen.flow_from_directory(train_path,
                                                      classes=['masks'],
                                                      color_mode = 'grayscale',
                                                      subset='validation',                                                    
                                                      **args)
    # Combine generators into one
    train_generator = zip((train_image_generator), (train_mask_generator))
    validation_generator = zip((validation_image_generator), (validation_mask_generator))

    callbacks_list = [
        # Adding checkpoints for the model 
        ModelCheckpoint(filepath = checkpoint_filepath, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True),
    ]

    # Fitting the model (params like epochs and steps_per_epoch were changing during the fitting, the fitting also could be stopped and then started by loading checkpoints)
    model.fit(train_generator,
              callbacks = callbacks_list,
              verbose = 1,
              steps_per_epoch = 500,
              epochs = 99,
              validation_data = validation_generator,
              validation_steps = 100,
              workers = 10)
    
    # Saving the model
    model.save('model1.h5')
    
if __name__ == '__main__':
    train()