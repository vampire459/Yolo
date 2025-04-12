import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# video_source = 0  # For webcam
# video_source = "https://media.istockphoto.com/id/1418377464/video/commuters-walking-to-work-back-view.mp4?s=mp4-640x640-is&k=20&c=YHy3Glq7naiWRBU57k1irATgHGV5q-6ZxHzIc4_QzU8="  # For video file
video_source = "https://b1a3526abf4b28.lhr.life/stream.html"  # For live stream (e.g., IP camera)

cap = cv2.VideoCapture(video_source)
# cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    person_detections = [det for det in results[0].boxes if det.cls == 0]

    person_count = len(person_detections)

    for box in person_detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf.item()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Persons detected: {person_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Crowd Counting", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
