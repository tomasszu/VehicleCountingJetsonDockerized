import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import time
import argparse
import torch


#============================================================================= <<<<<<<<<<<<<<<<<<<<<<<<<<
#Choose camera recording 1. to 3. or 5.
CAM = 1
#============================================================================= <<<<<<<<<<<<<<<<<<<<<<<<<<

def select_device():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA is available. Number of devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
        # Select the default device
        device = torch.device("cuda:0")
        print(f"Using device: {torch.cuda.get_device_name(device.index)}")
    else:
        print("CUDA is not available. Using CPU instead.")
        device = torch.device("cpu")
    
    return device

def printout(incount, outcount):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    F.write(("\n************Update at " + current_time + "*************\n"))
    F.write("Stats:" +
            "{\n" +
            "Total vehicles: "+ str(incount+outcount) + "\n" +
            "Vehicles inbound: "+ str(incount) + "\n" +
            "Vehicles outbound: "+ str(outcount) + "\n" +
            "}\n")
    F.flush()


def calculate_center(detection):
    x_center = (detection[0] + detection[2]) / 2.0
    y_center = (detection[1] + detection[3]) / 2.0
    #print(x_center, y_center)
    return [x_center, y_center]

def is_point_in_attention(point, vector1, vector2):

    if vector1 is not None and vector2 is not None:
        # Assuming vector1 and vector2 are represented by [x, y] points
        vector1, sign1 = vector1
        vector2, sign2 = vector2
        v1p1, v1p2 = vector1
        v2p1, v2p2 = vector2
        # Calculate the cross product to determine if the point is on the same side of the line
        if(sign1 == ">"):
            cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) > 0
        else:
            cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) < 0
        if(sign2 == ">"):
            cross_product2 = (v2p2[0]-v2p1[0])*(point[1] - v2p1[1]) - (v2p2[1] - v2p1[1]) *(point[0] - v2p1[0]) > 0
        else:
            cross_product2 = (v2p2[0]-v2p1[0])*(point[1] - v2p1[1]) - (v2p2[1] - v2p1[1]) *(point[0] - v2p1[0]) < 0
    elif(vector1 is not None):
        vector1, sign1 = vector1
        v1p1, v1p2 = vector1
        if(sign1 == ">"):
            cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) > 0
        else:
            cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) < 0
        cross_product2 = True
    else:
        cross_product = True
        cross_product2 = True

    cross_product = cross_product and cross_product2

    # If the cross product is positive, the point is on the same side as the frame
    return cross_product

def filter_detections(detections, vector_start, vector_end):
    indices_to_remove = []

    #print(detections)
    for i in range(len(detections.xyxy)):
        bbox_center = np.array([(detections.xyxy[i][0] + detections.xyxy[i][2]) / 2,
                                (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2])
        
        if not is_point_in_attention(bbox_center, vector_start, vector_end):
            indices_to_remove.append(i)

    # Remove unwanted detections in reverse order to avoid index issues
    for index in reversed(indices_to_remove):
        detections.xyxy = np.delete(detections.xyxy, index, axis=0)
        detections.confidence = np.delete(detections.confidence, index)
        detections.class_id = np.delete(detections.class_id, index)

    return detections

def detections_process(model, frame, tracker):
    confidence_threshold = 0.4

    results = model(frame)[0]
    #print(results.boxes)

    detections = sv.Detections.from_ultralytics(results)
    #print(detections)

    #mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections = detections[np.isin(detections.class_id, CLASS_ID)]
    detections = detections[np.greater(detections.confidence, confidence_threshold)]
    detections = filter_detections(detections, attention_vector1, attention_vector2)
    detections = tracker.update_with_detections(detections)

    COUNT_LINE.trigger(detections)

    return detections

def frame_annotations(detections, frame):

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    # format custom labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id, _
        in detections
    ]

    frame = LINE_ANNOTATOR.annotate(frame, COUNT_LINE)

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )


    annotated_labeled_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )


    annotated_labeled_frame = trace_annotator.annotate(
        scene=annotated_labeled_frame,
        detections=detections
    )

    return annotated_labeled_frame

if CAM == 1:
    start, end = sv.Point(x=-500, y=292), sv.Point(x=1878, y=292)
    attention_vector1 = [[0,175],[1279,175]], ">"
    attention_vector2 = [[0,505],[1140,0]], ">"
    cap = cv2.VideoCapture(f'cam{CAM}_cuts.mp4')
elif CAM == 2:
    start, end = sv.Point(x=-500, y=711), sv.Point(x=1878, y=198)
    cap = cv2.VideoCapture(f'cam{CAM}_cuts2.avi')
    attention_vector1 = [[0,120],[1279,570]], ">"
    attention_vector2 = [[63,0],[412,960]], "<"
elif CAM == 3:
    start, end = sv.Point(x=-500, y=600), sv.Point(x=1278, y=300)
    attention_vector1 = [[0,100],[2086,400]], ">"
    attention_vector2 = [[1500,0],[1900,2000]], ">"
    cap = cv2.VideoCapture(f'cam{CAM}_cuts.avi')
elif CAM == 5:
    start, end = sv.Point(x=1600, y=200), sv.Point(x=2600, y=2500)
    attention_vector1 = [[0,3000],[2000,0]], ">"
    attention_vector2 = None
    cap = cv2.VideoCapture(f'cam{CAM}_cuts2.avi')

MODEL = YOLO("yolov8n.pt")

# dict maping class_id to class_name
CLASS_NAMES_DICT = MODEL.model.names

# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

COUNT_LINE = sv.LineZone(start=start, end=end)

LINE_ANNOTATOR = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.5)

TRACKER = sv.ByteTrack()

F = open(f"output_files/output_cam{CAM}.txt", "w")

FOURCC = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'MJPG' or 'MP4V'

def main(output_mode,device):

    MODEL.to(device)

    curr_in_count = 0
    curr_out_count = 0

    ret, frame = cap.read()
    
    #out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 960))

    while ret:
    
        detections = detections_process(MODEL, frame, TRACKER)

        annotated_frame = frame_annotations(detections, frame)

        if (curr_in_count < COUNT_LINE.in_count or curr_out_count < COUNT_LINE.out_count):
            curr_in_count = COUNT_LINE.in_count
            curr_out_count = COUNT_LINE.out_count
            print("+1 vehicle has crossed")
            printout(curr_in_count,curr_out_count)
        #print(COUNT_LINE.in_count, COUNT_LINE.out_count)

        ## If we want to see video output (positional argument)
        if output_mode == 1:
            display = annotated_frame
            #out.write(display)
            display = cv2.resize(display, (1280, 960))

            cv2.imshow("Vehicle Detection", display)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        ret, frame = cap.read()
    
    cv2.destroyAllWindows()
    cap.release()
    F.close()

    #out.release()

if __name__ == "__main__":
    device = select_device()
    print(f"Selected device: {device}")

    parser = argparse.ArgumentParser(description="Vehicle counting in test video requires input 1 or 0 to show or not show video output during counting")
    parser.add_argument(
        "video_output",
        type=int,
        choices=[0, 1],
        help="Video output during testing: 1 for yes, 0 for no."
    )
    args = parser.parse_args()
    main(args.video_output,device)
