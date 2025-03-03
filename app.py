import cv2
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("runs/detect/train2/weights/best.pt")  # Use your trained model

def generate_frames():
    cap = cv2.VideoCapture(0)  # Change to 0 for webcam or provide video file path
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        for result in results:
            frame = result.plot()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
