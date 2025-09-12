import csv
import numpy as np
from neural_network.CNN import Network

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


def evaluate_label_fast(csv_path, model_path, target_label, false_check=False, false_samples=200, false_threshold=0.99):
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

    correct = np.sum(y_pred == y_true)
    print(f"\n📊 Точность по '{target_label}': {correct / len(y_true) * 100:.2f}% ({correct}/{len(y_true)})")

    # --- Проверка на ложных паттернах ---
    if false_check:
        false_X = generate_false_patterns(X, false_samples)
        probs = model.forward(false_X)[0][-1] if isinstance(model.forward(false_X), tuple) else model.forward(false_X)
        # Если forward возвращает softmax, то probs — это вероятности
        if probs.shape[0] != false_samples:
            # Если forward вернул (activations, pre_activations)
            probs = probs[-1]

        max_probs = np.max(probs, axis=1)
        safe_count = np.sum(max_probs < false_threshold)
        print(f"🧪 Ложные паттерны: {safe_count}/{false_samples} ({safe_count / false_samples * 100:.2f}%) "
              f"имели max_prob < {false_threshold}")

if __name__ == "__main__":
    for lbl in ["0", "1", "2", "3", "4", "5"]:
        evaluate_label_fast("../dataset/dataset.csv", "../models/model.csv", target_label=lbl, false_check=True)
