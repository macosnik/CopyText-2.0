import numpy as np
import csv
from CNN import Network

def generate_false_patterns(X, num_samples):
    false_patterns = []
    min_val, max_val = X.min(), X.max()

    for _ in range(num_samples):
        mode = np.random.choice([
            "random", "shuffle", "noise", "zero", "mix",
            "invert", "scale", "spike_noise", "block_zero",
            "class_mix", "perm_blocks", "gaussian_pattern"
        ])

        if mode == "random":
            vec = np.random.uniform(min_val, max_val, size=X.shape[1])

        elif mode == "shuffle":
            vec = X[np.random.randint(len(X))].copy()
            np.random.shuffle(vec)

        elif mode == "noise":
            vec = X[np.random.randint(len(X))].copy()
            vec += np.random.normal(0, 0.1, size=vec.shape)

        elif mode == "zero":
            vec = X[np.random.randint(len(X))].copy()
            mask = np.random.rand(len(vec)) < 0.3
            vec[mask] = 0

        elif mode == "mix":
            a = X[np.random.randint(len(X))]
            b = X[np.random.randint(len(X))]
            vec = (a + b) / 2

        elif mode == "invert":
            vec = max_val - X[np.random.randint(len(X))] + min_val

        elif mode == "scale":
            vec = X[np.random.randint(len(X))] * np.random.uniform(0.5, 1.5)

        elif mode == "spike_noise":
            vec = X[np.random.randint(len(X))].copy()
            spikes = np.random.rand(len(vec)) < 0.05
            vec[spikes] += np.random.uniform(-2, 2, size=spikes.sum())

        elif mode == "block_zero":
            vec = X[np.random.randint(len(X))].copy()
            block_size = np.random.randint(5, 20)
            start = np.random.randint(0, len(vec) - block_size)
            vec[start:start+block_size] = 0

        elif mode == "class_mix":
            samples = [X[np.random.randint(len(X))] for _ in range(np.random.randint(3, 6))]
            vec = np.mean(samples, axis=0)

        elif mode == "perm_blocks":
            vec = X[np.random.randint(len(X))].copy()
            block_size = np.random.randint(5, 20)
            blocks = [vec[i:i+block_size] for i in range(0, len(vec), block_size)]
            np.random.shuffle(blocks)
            vec = np.concatenate(blocks)

        elif mode == "gaussian_pattern":
            mean = np.random.uniform(min_val, max_val)
            std = np.random.uniform(0.05, 0.2)
            vec = np.random.normal(mean, std, size=X.shape[1])

        false_patterns.append(vec)

    return np.array(false_patterns, dtype=np.float32)

def load_dataset(csv_path, min_count=1000, false_ratio=0.2):
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

    # --- Подмешивание ложных паттернов ---
    num_false = int(len(X) * false_ratio)
    false_X = generate_false_patterns(X, num_false)
    uniform_Y = np.full((num_false, len(valid_labels)), 1.0 / len(valid_labels), dtype=np.float32)

    X = np.vstack([X, false_X])
    Y = np.vstack([Y, uniform_Y])

    print(f"➕ Добавлено {num_false} ложных паттернов с равномерными метками")
    print(f"\n📦 Итоговый датасет: X.shape={X.shape}, Y.shape={Y.shape} (features={num_features})\n")
    return X, Y, valid_labels

def train_cnn(csv_path, save_path, min_count, lr, max_epochs, val_ratio, false_ratio=0.2):
    X, Y, labels = load_dataset(csv_path, min_count=min_count, false_ratio=false_ratio)

    y_indices = np.argmax(Y, axis=1)

    val_size = int(len(X) * val_ratio)

    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y_indices[:-val_size], y_indices[-val_size:]

    model = Network(
        layers=[X.shape[1], 256, 128, len(labels)],
        learning_rate=lr
    )

    model.train(X_train, y_train, X_val, y_val, max_epochs, save_path)
    model.save_model_csv(save_path)

if __name__ == "__main__":
    train_cnn("../dataset/dataset.csv", "../models/model.csv", 1000, 0.01, 10000, 0.5, false_ratio=0.2)
