import csv
import json
import numpy as np
from colorama import init, Style

init()

DATASET_FILE = 'dataset.csv'
NAMES_FILE = 'settings.json'

def remove_empty_images(file, thr=0.0):
    import csv, numpy as np
    with open(file) as f:
        h, *d = csv.reader(f)
    d = [r for r in d if not np.all(np.array(r[:-1], float) <= thr)]
    with open(file, 'w', newline='') as f:
        csv.writer(f).writerows([h] + d)

remove_empty_images(DATASET_FILE)

with open(NAMES_FILE, encoding='utf-8') as f:
    names_data = json.load(f)
SIZE = int(names_data.get("size"))

def gray_to_block(val):
    gray_level = int(val * 23) + 232
    return f"\033[48;5;{gray_level}m   {Style.RESET_ALL}"

with open(DATASET_FILE, newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = list(reader)

labels = sorted(set(row[-1] for row in data))
print("Доступные метки:")
for lbl in labels:
    count = sum(1 for row in data if row[-1] == lbl)
    print(f"- {lbl} ({count})")

choice = input("\nВведите имя метки: ").strip()

filtered = [(idx, row) for idx, row in enumerate(data) if row[-1] == choice]

if not filtered:
    print("Нет изображений с такой меткой.")
    exit()

deleted_indices = []
for idx, row in filtered:
    pixels = np.array(row[:-1], dtype=float).reshape((SIZE, SIZE))
    label = row[-1]
    print(f"\n\033[96mИзображение {idx+1}, метка: {label}{Style.RESET_ALL}")
    for y in range(SIZE):
        print(''.join(gray_to_block(p) for p in pixels[y]))

    action = input("Нажмите Enter для следующего или введите 'delete' для удаления: ").strip().lower()
    if action == "delete":
        deleted_indices.append(idx)

if deleted_indices:
    data = [row for i, row in enumerate(data) if i not in deleted_indices]
    with open(DATASET_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"Удалено {len(deleted_indices)} примеров.")

print("Готово.")
