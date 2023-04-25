import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report


def validate(excel_path):
    # read excel file
    file = pd.read_excel(excel_path)
    cm = pd.crosstab(file.Pred_Label, file.True_Label)
    fig = plt.figure()
    ax1 = plt.figure(1, 2, 1)
    sn.heatmap(cm, annot=True, cmap='Blues')
    for i in range(cm.shape[1]):
        TP = cm.iloc[i, i]
        FP = cm.iloc[i, :].sum() - TP
        FN = cm.iloc[:, i].sum() - TP
        TN = cm.sum().sum() - TP - FP - FN
        Acc = (TP + TN) / cm.sum().sum()
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        try:
            F1_score = (2 * Precision * Recall) / (Precision + Recall)
        except Exception:
            F1_score = 0
        print(cm.index[i], Acc, Precision, Recall, F1_score)
    df = pd.DataFrame(classification_report(file.True_Label, file.Pred_Label, output_dict=True, zero_division=0)).T
    df.to_excel("./AgriNer_Code/validation.xlsx")
    print("Validation excel file has been generated /AgriNer_Code/validation.xlsx")
    return "./AgriNer_Code/validation.xlsx"