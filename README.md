# 项目介绍
...
您可以在我们的[网站](http://47.94.162.105/model1/)上方便地使用已部署好的最佳模型。

# 安装
代码在`python 3.9`环境下运行，请按照以下步骤安装
```bash
git clone https://github.com/winterwinds/ITP-Gut-Microbiota.git
cd ITP-Gut-Microbiota

conda env create -f requirements.yaml
conda activate ITPGM
```
如果您使用`pip`
```bash
pip install numpy pandas seaborn matplotlib scikit-learn openpyxl jupyter
```

# 使用指南
1. 生成最佳模型
运行 `main.py`，生成最佳模型（默认存储在 `trained_models` 文件夹中）及其阈值下的 *MCC* 值变化曲线（默认存储在 `mcc_threshold_curve` 文件夹中）：
```bash
python main.py --n_splits 5 --min_thres 0.1 --max_thres 0.9 --interval 0.1
```

可选参数：
* --n_splits: 使用的交叉验证折数（默认为 5）
* --min_thres: 阈值范围的最小值（默认为 0.1）
* --max_thres: 阈值范围的最大值（默认为 0.9）
* --interval: 阈值的变化步长（默认为 0.1）
* --data: 指定自定义数据集的路径，需符合 src/data_processing/dataset.py 的格式要求。

2. 测试模型性能
使用 `eval.py` 测试指定模型在测试集上的性能，传入模型路径和指定的阈值：
```bash
python eval.py --model_path "trained_models/SVM_(RBF).pkl" --threshold 0.6
```

输出包括：

* 测试集上的 `MCC`、`AUC`、`F1` 等指标
* 混淆矩阵结果

# License
本项目基于 MIT License 开源，详情请参考 LICENSE 文件。
