import csv
import numpy as np

from neural_network.CNN import Network

def evaluate_label_fast(csv_path, model_path, target_label):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        feature_columns = [col for col in reader.fieldnames if col != "label"]

    all_labels = sorted(set(row["label"] for row in rows))

    if target_label not in all_labels:
        print(f"❌ Метка '{target_label}' не найдена. Доступные: {all_labels}")
        return

    target_idx = all_labels.index(target_label)

    X = []
    y_true = []
    for row in rows:
        if row["label"] == target_label:
            X.append([float(row[col]) for col in feature_columns])
            y_true.append(target_idx)

    X = np.array(X, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    model = Network(layers=[])
    model.load_model_csv(model_path)

    y_pred = model.predict(X)

    table = []
    correct = 0
    for idx, (true_idx, pred_idx) in enumerate(zip(y_true, y_pred), start=1):
        status = "✅" if true_idx == pred_idx else "❌"
        if status == "✅":
            correct += 1
        table.append([
            idx,
            all_labels[true_idx],
            all_labels[pred_idx],
            status
        ])

    print(f"\n📊 Точность по '{target_label}': {correct / len(y_true) * 100:.2f}% ({correct}/{len(y_true)})")

if __name__ == "__main__":
    evaluate_label_fast("../dataset/dataset.csv", "../models/model.csv", target_label="1")
