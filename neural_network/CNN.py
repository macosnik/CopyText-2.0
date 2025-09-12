import numpy as np
import time
import csv

class Network:
    def __init__(self, layers, learning_rate=0.01, min_lr=1e-5):
        self.layers = layers
        self.lr = learning_rate
        self.min_lr = min_lr
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = len(y_true)
        y_onehot = np.zeros_like(y_pred)
        y_onehot[np.arange(m), y_true] = 1
        return -np.sum(y_onehot * np.log(y_pred + 1e-9)) / m

    def forward(self, X):
        activations = [X]
        pre_activations = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = self.relu(z)
            activations.append(a)

        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        a = self.softmax(z)
        activations.append(a)

        return activations, pre_activations

    def backward(self, activations, pre_activations, y_true):
        grads_w = []
        grads_b = []

        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(len(y_true)), y_true] = 1

        delta = activations[-1] - y_onehot

        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            dw = a_prev.T @ delta / len(y_true)
            db = np.sum(delta, axis=0, keepdims=True) / len(y_true)

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i != 0:
                delta = (delta @ self.weights[i].T) * self.relu_deriv(pre_activations[i-1])

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def train_step(self, X, y):
        activations, pre_activations = self.forward(X)
        self.backward(activations, pre_activations, y)

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def save_model_csv(self, filename):
        with open(filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "type", "i", "j", "value"])

            for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        writer.writerow([layer_idx, "W", i, j, w[i, j]])

                for j in range(b.shape[1]):
                    writer.writerow([layer_idx, "B", 0, j, b[0, j]])

    def load_model_csv(self, filename):
        temp_weights = {}
        temp_biases = {}

        with open(filename, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer_idx = int(row["layer"])
                typ = row["type"]
                i = int(row["i"])
                j = int(row["j"])
                val = float(row["value"])

                if typ == "W":
                    if layer_idx not in temp_weights:
                        temp_weights[layer_idx] = {}
                    temp_weights[layer_idx][(i, j)] = val
                elif typ == "B":
                    if layer_idx not in temp_biases:
                        temp_biases[layer_idx] = {}
                    temp_biases[layer_idx][(i, j)] = val

        layers = []
        sorted_layers = sorted(temp_weights.keys())
        for idx in sorted_layers:
            rows = max(i for (i, _) in temp_weights[idx].keys()) + 1
            cols = max(j for (_, j) in temp_weights[idx].keys()) + 1
            if idx == 0:
                layers.append(rows)
            layers.append(cols)

        self.layers = layers
        self.weights = []
        self.biases = []

        for idx in sorted_layers:
            rows = max(i for (i, _) in temp_weights[idx].keys()) + 1
            cols = max(j for (_, j) in temp_weights[idx].keys()) + 1

            w = np.zeros((rows, cols), dtype=np.float32)
            for (i, j), val in temp_weights[idx].items():
                w[i, j] = val

            b = np.zeros((1, cols), dtype=np.float32)
            for (i, j), val in temp_biases[idx].items():
                b[i, j] = val

            self.weights.append(w)
            self.biases.append(b)

    def train(self, X_train, y_train, X_val, y_val, max_epochs, save_path, batch_size=64, shuffle=True):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        start_time = time.time()

        n_samples = X_train.shape[0]

        for epoch in range(1, max_epochs + 1):
            # Перемешиваем данные в начале каждой эпохи
            if shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X_train = X_train[indices]
                y_train = y_train[indices]

            # --- Обучение по батчам ---
            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                self.train_step(X_batch, y_batch)

            # --- Оценка после эпохи ---
            train_pred, _ = self.forward(X_train)
            val_pred, _ = self.forward(X_val)

            train_loss = self.cross_entropy_loss(train_pred[-1], y_train)
            val_loss = self.cross_entropy_loss(val_pred[-1], y_val)

            print(f"\rEpoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {self.lr:.6f}",
                  end="")

            # Сохраняем лучшую модель
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_model_csv(save_path)
            else:
                epochs_no_improve += 1

            # Early stopping / снижение LR
            if epochs_no_improve >= 5:
                if self.lr > self.min_lr:
                    self.lr = max(self.lr / 10, self.min_lr)
                    print(f"\n\n⚠ Learning rate reduced to {self.lr}", end="")
                    epochs_no_improve = 0
                else:
                    print("\n\n⛔ Early stopping: no improvement and min LR reached", end="")
                    break

        # --- Итог ---
        end_time = time.time()
        elapsed = end_time - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print("\n\n📊 Обучение завершено")
        print(f"⏱ Время обучения: {hours}ч {minutes}м {seconds}с")
        print(f"Лучший val_loss: {best_val_loss:.4f}")
        print(f"Финальный LR: {self.lr:.6f}")

