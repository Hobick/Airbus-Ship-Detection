1. data.py file scanning for all images in train csv file and generating masks using the mask_decode function saving these masks. Images without hte ships are deleting.
2. unet_model.py contains th U-net architecture based neural network in function unet()
3. train.py creates the model using unet() func (and loading weigths) then creates 2 generators either for training and validating data and then fitting useing these generators
4. predict.py loads the model, reding the csv file with test data and generating masks saving it into data/results/
5. submission.py generates csv submission file for kaggle competition
