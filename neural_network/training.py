import numpy as np
import csv

from CNN import Network

def load_dataset(csv_path, min_count=1000):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
        all_columns = reader.fieldnames

    feature_columns = [col for col in all_columns if col != "label"]
    num_features = len(feature_columns)

    labels = np.array([row["label"] for row in data])
    unique_labels, counts = np.unique(labels, return_counts=True)

    print("📊 Количество паттернов по меткам:")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"  {lbl}: {cnt}")

    valid_labels = unique_labels[counts >= min_count].tolist()
    print(f"\n✅ Метки, прошедшие фильтр (>= {min_count}): {valid_labels}")

    label_to_samples = {}
    for lbl in valid_labels:
        lbl_samples = [row for row in data if row["label"] == lbl]
        idx = np.random.choice(len(lbl_samples), size=min_count, replace=False)
        label_to_samples[lbl] = [lbl_samples[i] for i in idx]

    X, Y = [], []
    label_to_index = {lbl: idx for idx, lbl in enumerate(valid_labels)}

    for i in range(min_count):
        for lbl in valid_labels:
            row = label_to_samples[lbl][i]
            features = [float(row[col]) for col in feature_columns]
            X.append(features)

            one_hot = [0] * len(valid_labels)
            one_hot[label_to_index[lbl]] = 1
            Y.append(one_hot)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"\n📦 Итоговый датасет: X.shape={X.shape}, Y.shape={Y.shape} (features={num_features})\n")
    return X, Y, valid_labels

def train_cnn(csv_path, save_path, min_count, lr, max_epochs, val_ratio):
    X, Y, labels = load_dataset(csv_path, min_count=min_count)

    y_indices = np.argmax(Y, axis=1)

    val_size = int(len(X) * val_ratio)

    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y_indices[:-val_size], y_indices[-val_size:]

    model = Network(
        layers=[X.shape[1], 128, len(labels)],
        learning_rate=lr
    )

    model.train(X_train, y_train, X_val, y_val, max_epochs, save_path)
    model.save_model_csv(save_path)

if __name__ == "__main__":
    train_cnn("../dataset/dataset.csv", "../models/model.csv", 100, 0.01, 1000, 0.2)
