import cv2
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train2/weights/best.pt")  # Update with your best model path

# Open the webcam (use '0' for default webcam, or replace with your RTSP stream URL)
cap = cv2.VideoCapture(0)  # Change to "rtsp://your_camera_ip" for an IP camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame)

    # Render results on the frame
    for result in results:
        frame = result.plot()

    # Display the frame with detections
    cv2.imshow("Shoplifting Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
