from flask import Flask, request, render_template, Response, jsonify
import cv2
import os
import torch
import tempfile
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/detect/train2/weights/best.pt', force_reload=True)

# Ensure temp directory exists
temp_dir = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        # Perform detection
        results = model(filepath)
        results.save(temp_dir)  # Save result images in temp directory

        return jsonify({'message': 'Detection completed', 'file': filename})

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        out_filename = os.path.join(temp_dir, f'detected_{filename}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_filename, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for box in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        return jsonify({'message': 'Video detection completed', 'file': f'detected_{filename}'})

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'running'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
