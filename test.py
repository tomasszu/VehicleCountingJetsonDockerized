import cv2

from ultralytics import YOLO

# Load an official or custom model
model = YOLO("yolov8n.pt")  # Load an official Detect model

# Perform tracking with the model
# results = model.track("/home/tomass/tomass/Vehicle_counting_it2_test/cam1_cuts2.avi", show=True)  # Tracking with default tracker
# results = model.track("/home/tomass/tomass/Vehicle_counting_it2_test/cam1_cuts2.avi", conf=0.8, iou=0.5, show=True, tracker="bytetrack.yaml")  # with ByteTrack

# Open the video file
video_path = "/home/tomass/tomass/Vehicle_counting_it2_test/cam1_cuts2.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.5, iou=0.4, tracker="bytetrack.yaml", persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()