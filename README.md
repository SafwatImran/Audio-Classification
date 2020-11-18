# Audio-Classification
# Both models were tested using the Urbansound8k, ESC-50 and FSDKaggle 2018 datasets.
# In order to run the cf model, directly allow the parser to create the needed matrices by loading the appropriate files from a dataset that is divided into folds and categories given as classID's. Change the directory to load the csv file to a relative path beforehand.
# In order to run the cfclean model, modify the dataset csv file to 'fname' as the filename column and 'label' as the category. Keep all files of the dataset in the wavfiles directory. Run the eda file, to obtain 16khz samples in the clean directory and then proceeed to run the model from model.py. 
