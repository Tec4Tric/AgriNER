import json
import spacy
import pandas  as pd
def predict(model_path, test_Data):
    nlp_ner = spacy.load(model_path)
    f = open(test_Data, encoding='utf-8')
    test_data = json.load(f)
    def get_true_label(entity, curToken):
        loc =''
        for ent in entity:
            if curToken in ent[-1]:
                loc =  ent[-2]
        return loc

    total_entity=[]
    true_class=[]
    pred_class=[]
    for annot in test_data:
        docs = nlp_ner(annot['text'])
        ent = list(docs.ents)
        total_entity.append(ent)
        curRef = 0
        for i,tok in enumerate(docs):
            curToken = (ent[curRef].text).split(" ")[-1]
            if tok.text == curToken:
                entity = annot['entities']
                a = get_true_label(entity, curToken)
                total_entity.append(ent[curRef])
                true_class.append(a)
                pred_class.append(tok.ent_type_)
                curRef += 1
                if curRef == len(ent):
                    break

    df = pd.DataFrame.from_dict({ 'True_Label': true_class, 'Pred_Label': pred_class})
    df.to_excel('./AgriNer_Code/Prediction_output.xlsx')
    print("Output excel file has been generated AgriNer_Code/Prediction_output.xlsx")
    return "./AgriNer_Code/Prediction_output.xlsx"

