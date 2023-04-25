import os
from data_preprocess import run
from split_data import data_splitter
from train_custom_NER import train
from predict import predict
from validation import validate

newpath = r'./AgriNer_Code' # It will create a folder named "AgriNer_Code", where the code will produce avery output files
if not os.path.exists(newpath):
    os.makedirs(newpath)
path = "/home/raj/brat/data/dataset"

''' The below function takes the folder location of the annotation file, where user have the ".txt" file and ".ann" file. They can have, multiple files or single file. As an ouput, this will return a processed JSON file. '''
processed_data = run(path)


''' The below function will split the dataset with custom ratio, if not defined, it will split the dataset in to 70:30 ratio.
First define train, then test variavle to store train and test data. '''
train_data, test_data = data_splitter(processed_data)

''' The below function will take the train and test file and will create train and test spacy file '''
outut = train(train_data, test_data)

'''  Open terminal in the same working directory, and run the following code, one by one. '''
''' python -m spacy init fill-config ./AgriNer/conf_files/base_config.cfg ./AgriNer/conf_files/config.cfg '''
''' python -m spacy train ./AgriNer/conf_files/config.cfg --output ./AgriNer/AgriNer_Code --paths.train ./AgriNer/AgriNer_Code/train.spacy --paths.dev ./AgriNer/AgriNer_Code/train.spacy '''
model_path = "/home/raj/NLP-Code/Test/NLP/NEW/model-last"
pred_output = predict(model_path, test_data)
validate = validate(pred_output)


''' ---------------- XXXXXXXXXXXXXXXXXXXXXXXXX -----------------------'''