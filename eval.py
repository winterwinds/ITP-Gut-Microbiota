from src.data_processing import FeatureManager
from src.data_processing import Dataset
from src.optimization.model_selection import best_model_by_auc, threshold_by_mcc
from src.model import plot_mcc_variation, test_metrics
import argparse
import os
import pickle

def main():
    default_data_file = os.path.join("data","processed","cleaned_data.csv")

    parser = argparse.ArgumentParser(description="ITPGM Model Selection and Evaluation")
    
    parser.add_argument("--data", type=str, default=default_data_file, help="Path to the dataset (CSV or XLSX file).")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold")

    args = parser.parse_args()
    fm = FeatureManager().get_all_features()
    dataset = Dataset(args.data, fm, 'group', {'R':1, 'NR': 0},'type')

    metrics, confusion = test_metrics(args.model_path, dataset.X_test, dataset.y_test, args.threshold)
    print(f"Metrics of {args.model_path}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print(f"Confusion of {args.model_path}:")
    print(confusion)

if __name__ == "__main__":
    main()