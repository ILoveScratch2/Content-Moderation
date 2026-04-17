"""
DistilBERT 文本分类推理脚本
用法:
  单条: python predict.py --model ./best_model --text "待检测文本"
  批量: python predict.py --model ./best_model --input test.csv --output results.csv
"""

import argparse

import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


class Predictor:
    def __init__(self, model_dir: str):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=256
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        return {
            "text": text,
            "label": pred,
            "confidence": round(probs[0][pred].item(), 4),
            "result": "敏感" if pred == 1 else "正常",
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            for j, (text, pred) in enumerate(zip(batch, preds)):
                p = pred.item()
                results.append(
                    {
                        "text": text,
                        "label": p,
                        "confidence": round(probs[j][p].item(), 4),
                        "result": "敏感" if p == 1 else "正常",
                    }
                )
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./best_model")
    parser.add_argument("--text", type=str, help="单条文本预测")
    parser.add_argument("--input", type=str, help="批量预测的 CSV 文件")
    parser.add_argument("--output", type=str, default="results.csv")
    args = parser.parse_args()

    predictor = Predictor(args.model)

    if args.text:
        result = predictor.predict(args.text)
        print(f"文本: {result['text']}")
        print(f"结果: {result['result']} (置信度: {result['confidence']})")
    elif args.input:
        df = pd.read_csv(args.input)
        results = predictor.predict_batch(df["text"].tolist())
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"结果已保存到 {args.output}, 共 {len(results)} 条")
    else:
        print("请指定 --text 或 --input 参数")


if __name__ == "__main__":
    main()
