import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve, auc, f1_score
import numpy as np
import json
from src.data_processing import Dataset
from src.data_processing import FeatureManager
import os
import pickle

from .search_config import param_grids
from .search_config import classifiers

def best_model_by_auc(X_train, y_train, n_split, random_state=42):

    selected_models = []
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=random_state)

    for clf_name, param_grid in param_grids.items():
        print(f"Searching best params for: {clf_name}")
        pipeline = Pipeline([("scale", StandardScaler()), ("classifier", classifiers[clf_name])])
        grid_search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=skf, scoring='roc_auc', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_auc = grid_search.best_score_
        print(f"The best_auc of {clf_name}: {best_auc:.3f}")

        model_file_path = os.path.join("trained_models", f"{clf_name.replace(' ','_')}.pkl")
        with open(model_file_path, "wb") as f:
            pickle.dump(grid_search.best_estimator_, f)
        print(f"Model {clf_name} saved at: {model_file_path}")
        
        selected_models.append((clf_name, grid_search.best_estimator_, best_auc))

    return selected_models

def best_threshold_by_mcc(modelfile, X_train, y_train, n_split, min_thres, max_thres, interval):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    thresholds = np.linspace(min_thres, max_thres, num=int((max_thres - min_thres) / interval) + 1)
    thresholds = np.round(thresholds, decimals=3)

    with open(modelfile,"rb") as f:
        model = pickle.load(f)

    mcc_scores = {threshold: [] for threshold in thresholds}

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            if len(np.unique(y_pred)) < 2:
                #print(f"threshold {threshold:.2f} predicting all the same, pass.")
                continue
            mcc = matthews_corrcoef(y_val, y_pred)
            mcc_scores[threshold].append(mcc)

    avg_mcc_scores = {
        threshold: np.mean(scores) for threshold, scores in mcc_scores.items() if scores
    }
    if not avg_mcc_scores:
        print(f"Model has no valid mcc.")

    return avg_mcc_scores


def threshold_by_mcc(selected_models, X_train, y_train, n_split, min_thres, max_thres, interval):

    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    thresholds = np.linspace(min_thres, max_thres, num=int((max_thres - min_thres) / interval) + 1)
    thresholds = np.round(thresholds, decimals=3)
    final_results = []
    mcc_variation = {}

    for clf_name, model, auc_score in selected_models:

        print(f"Searching for best threshold: {clf_name}")
        mcc_scores = {threshold: [] for threshold in thresholds}

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]

            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                if len(np.unique(y_pred)) < 2:
                    #print(f"{clf_name}'s threshold {threshold:.2f} predicting all the same, pass.")
                    continue
                mcc = matthews_corrcoef(y_val, y_pred)
                mcc_scores[threshold].append(mcc)

        avg_mcc_scores = {
            threshold: np.mean(scores) for threshold, scores in mcc_scores.items() if scores
        }
        if not avg_mcc_scores:
            print(f"Model {clf_name} has no valid mcc, pass.")
            continue

        best_threshold = max(avg_mcc_scores, key=avg_mcc_scores.get)
        best_mcc = avg_mcc_scores[best_threshold]

        mcc_variation[clf_name] = avg_mcc_scores

        final_results.append({
            "Model": clf_name,
            "AUC": auc_score,
            "Best MCC": best_mcc,
            "Best Threshold": best_threshold
        })

    final_results_df = pd.DataFrame(final_results)
    final_results_df.to_excel("Final_Model_Selection.xlsx", index=False)
    print("Result saved in file.")

    with open("MCC_Variation.json", "w") as f:
        json.dump(mcc_variation, f)

    return mcc_variation


if __name__ == '__main__':
    fm = FeatureManager().get_all_features()
    dataset = Dataset('data/processed/cleaned_data.csv', fm, 'group', {'R': 1, 'NR':0}, 'type')

    models = best_model_by_auc(dataset.X_train, dataset.y_train, 5, 42)
    threshold_by_mcc(models, dataset.X_train, dataset.y_train, 5, 0.1, 0.9, 0.1)
