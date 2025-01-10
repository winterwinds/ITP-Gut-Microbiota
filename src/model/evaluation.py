import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)
import json
import os
import pickle

def plot_mcc_variation(mcc_variation_file):

    with open(mcc_variation_file,"r") as f:
        mcc_variation = json.load(f)
    
    for clf_name, mcc_scores in mcc_variation.items():
        mcc_scores = {float(threshold): score for threshold, score in mcc_scores.items()}
        thresholds, mccs = zip(*sorted(mcc_scores.items()))
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=thresholds, y=mccs, marker='o')
        plt.title(f"MCC vs Threshold for {clf_name}")
        plt.xlabel("Threshold")
        plt.ylabel("MCC")
        plt.grid()

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))

        save_path = os.path.join("mcc_threshold_curve", f"MCC_Threshold_{clf_name}.png")
        plt.savefig(save_path)
        print(f"曲线图已保存至: {save_path}")

def test_metrics(model_path, X_test, y_test, threshold):

    with open(model_path,"rb") as f:
        model = pickle.load(f)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    confusion = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "MCC": mcc,
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy
    }
    
    return metrics, confusion
