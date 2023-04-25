import os
import re
import json
import pandas as pd


def get_entity(str, text_ann):
    relation =  re.search(str + ",([\w]*)", text_ann).group(1)
    start = re.search(str + ",([\w]*)\s([\d\s]*)", text_ann).group(2).split(" ")[0]
    end = re.search(str + ",([\w]*)\s([\d\s]*)", text_ann).group(2).split(" ")[1]
    return relation, start, end

def get_data(paths):
    paths = paths if paths.endswith("/") else paths+"/"
    temp_dict = {}
    temp_dict['entities'] = []
    temp_dict['relations'] = []
    temp_dict['text'] = ""
    count_dict={}
    training_data = []
    total_len = 0
    for i in os.listdir(paths):
        temp_dict = {}
        temp_dict['entities'] = []
        temp_dict['relations'] = []
        temp_dict['text'] = ""
        temp_dict['file'] = ""
        if i.endswith(".ann") or i.endswith('.stats_cache'):
            continue
        else:
            filename, extn = os.path.splitext(i)
            if extn.endswith(".ann") or extn.endswith(".txt"):
                text_str = open(paths+filename+".txt",encoding='utf-8').read()
                text_ann = open(paths+filename+".ann", encoding='utf-8').read().replace("\t", ",")
            else:
                continue
            for data in text_ann.split('\n'):
                if data.split(",")[0].startswith("R"):
                    relation = data.split(",")[1].split(" ")[0]
                    agr1 = data.split(",")[1].split(" ")[1].split(":")[1]
                    agr2 = data.split(",")[1].split(" ")[2].split(":")[1]
                    entity1, start1, end1 = get_entity(agr1, text_ann)
                    entity2, start2, end2 = get_entity(agr2, text_ann)

                    temp_dict['relations'].append([int(start1), int(end1), int(start2), int(end2), relation])
                if data.split(",")[0].startswith("T"):
                    start = data.split(",")[1].split(" ")[1]
                    end = data.split(",")[1].split(" ")[2]
                    label = data.split()[0].split(",")[1]
                    span = data.split(",")[-1]
                    temp_dict['entities'].append((int(start) + total_len, int(end) + total_len, label, span))
                    try:
                        if count_dict[label] == "":
                            count_dict[label] = 0
                        else:
                            count_dict[label] = count_dict[label] + 1
                    except Exception:
                        count_dict[label] = 1
            temp_dict['text']+= text_str
            temp_dict['file'] = filename+".txt"
        training_data.append(temp_dict)
    return training_data, count_dict
def run(paths):
    training_data, count_dict = get_data(paths)
    with open("./AgriNer_Code/data.json", "w") as outfile:
        json.dump(training_data, outfile)
    # Saving the count number into an excel
    df = pd.DataFrame(data=count_dict, index=[0])
    df = (df.T)
    df.to_excel('./AgriNer_Code/count.xlsx')
    print("JSON and Excel file has been created successfully in AgriNer_Code folder.")
    return './AgriNer_Code/data.json'


