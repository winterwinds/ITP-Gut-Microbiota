from src.data_processing import FeatureManager
from src.data_processing import Dataset
from src.optimization.model_selection import best_model_by_auc, threshold_by_mcc
from src.model import plot_mcc_variation, test_metrics
import argparse
import os


def main():
    default_data_file = os.path.join("data","processed","cleaned_data.csv")

    parser = argparse.ArgumentParser(description="ITPGM Model Selection and Evaluation")
    
    parser.add_argument("--data", type=str, default=default_data_file, help="Path to the dataset (CSV or XLSX file).")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for cross-validation.")
    parser.add_argument("--min_thres", type=float, default=0.1, help="Minimum threshold for MCC optimization.")
    parser.add_argument("--max_thres", type=float, default=0.9, help="Maximum threshold for MCC optimization.")
    parser.add_argument("--interval", type=float, default=0.1, help="Interval for threshold optimization.")
    
    args = parser.parse_args()
    fm = FeatureManager().get_all_features()
    dataset = Dataset(args.data, fm, 'group', {'R':1, 'NR': 0},'type')

    print("Dataset loaded successfully.")
    print("——————————————————————————————————")
    print("Features: ")
    print(fm)
    print("——————————————————————————————————")

    models = best_model_by_auc(dataset.X_train, dataset.y_train, args.n_splits)
    print("——————————————————————————————————")

    mcc_variation = threshold_by_mcc(
        models, dataset.X_train, dataset.y_train, args.n_splits,
        args.min_thres, args.max_thres, args.interval
    )
    print("——————————————————————————————————")

    plot_mcc_variation("MCC_Variation.json")

if __name__ == "__main__":
    main()