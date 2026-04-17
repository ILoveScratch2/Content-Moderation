# DistilBERT 文本分类器

## 项目简介
本项目基于 DistilBERT 实现了一个轻量级的文本分类器，支持文本分类任务。适用于敏感内容检测。

## 功能
- 训练脚本：使用 `train.py` 训练模型，支持自定义数据集和超参数。
- 推理脚本：使用 `predict.py` 进行单条或批量文本分类。


## 使用方法

安装依赖：
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python train.py --data train.csv --output ./best_model
```
- `--data`：训练数据的 CSV 文件，需包含 `text` 和 `label` 两列。
- `--output`：模型保存路径。

### 推理
#### 单条文本预测
```bash
python predict.py --model ./best_model --text "待检测文本"
```
#### 批量文本预测
```bash
python predict.py --model ./best_model --input test.csv --output results.csv
```
- `--input`：待预测的 CSV 文件，需包含 `text` 列。
- `--output`：预测结果保存路径。

## 数据格式
训练和预测所需的 CSV 文件需满足以下格式：
```csv
text,label
示例文本1,0
示例文本2,1
```
- `text`：文本内容。
- `label`：标签，0 表示正常，1 表示敏感。

