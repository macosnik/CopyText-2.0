from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import csv, os, io, base64, re, json
import numpy as np
import threading, queue

app = Flask(__name__)

DATASET_FILE = 'dataset.csv'
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

with open(SETTINGS_FILE, encoding='utf-8') as f:
    EXPORT_SIZE = int(json.load(f).get("size", 28))

save_queue = queue.Queue()

label_counts = {}

if os.path.isfile(DATASET_FILE):
    with open(DATASET_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row['label']
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

def save_worker():
    while True:
        data = save_queue.get()
        if data is None:
            break
        try:
            img = crop_and_resize(data_url_to_image(data['image']), EXPORT_SIZE)
            arr = 1.0 - (np.array(img, dtype=np.float32) / 255.0)
            row = [f"{v:.6f}" for v in arr.flatten()] + [data['label']]
            file_exists = os.path.isfile(DATASET_FILE)
            with open(DATASET_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([f'p{i}' for i in range(EXPORT_SIZE**2)] + ['label'])
                writer.writerow(row)
        except Exception as e:
            print("Ошибка сохранения:", e)
        finally:
            save_queue.task_done()

threading.Thread(target=save_worker, daemon=True).start()

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)

@app.route('/count/<label>')
def count_label(label):
    return jsonify({'count': label_counts.get(label, 0)})

def data_url_to_image(data_url: str) -> Image.Image:
    b64data = re.sub(r"^data:image/(png|jpeg);base64,", "", data_url)
    return Image.open(io.BytesIO(base64.b64decode(b64data))).convert("L")

def crop_and_resize(img: Image.Image, size: int) -> Image.Image:
    arr = np.array(img)
    coords = np.argwhere(arr < 255)
    if coords.size == 0:
        return Image.new("L", (size, size), 255)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = img.crop((x0, y0, x1, y1))
    max_side = max(cropped.size)
    square = Image.new("L", (max_side, max_side), 255)
    square.paste(cropped, ((max_side - cropped.size[0]) // 2,
                           (max_side - cropped.size[1]) // 2))
    return square.resize((size, size), Image.LANCZOS)

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    if not data.get('image') or data.get('label') is None:
        return jsonify({'status': 'error', 'message': 'invalid data'}), 400

    label = data['label']

    label_counts[label] = label_counts.get(label, 0) + 1

    save_queue.put(data)

    return jsonify({'status': 'ok', 'count': label_counts[label]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050)
