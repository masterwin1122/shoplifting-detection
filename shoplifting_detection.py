import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Ensure the model file is downloaded

# Initialize MediaPipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the webcam (or CCTV camera feed)
cap = cv2.VideoCapture(0)  # Change to RTSP link for CCTV

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object Detection with YOLOv8
    results = model(frame)
    
    # Convert image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)

    # Draw YOLO bounding boxes
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Shoplifting Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
