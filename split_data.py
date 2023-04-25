import json
import random

def data_splitter(path,split_size=0.7):
    f = open(path)
    train_data = []
    test_data = []
    training_data = json.load(f)
    random.shuffle(training_data)
    for i,data in enumerate(training_data):
        if i < len(training_data)*split_size:
            train_data.append(data)
        else:
            test_data.append(data)
    with open("./AgriNer_Code/train_data.json", "w") as train_file:
        json.dump(train_data, train_file)
    with open("./AgriNer_Code/test_data.json", "w") as test_file:
        json.dump(test_data, test_file)
    print("Data splitted with {}:{} ratio.".format(int(split_size*100), int(100-(split_size*100))))
    return "./AgriNer_Code/train_data.json", "./AgriNer_Code/test_data.json"
