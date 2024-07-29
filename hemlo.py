import cv2
import numpy as np
from playsound import playsound

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture('one.mp4')

# Function to detect vehicles
def detect_vehicles(frame):
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # class_id == 2 is for car in COCO dataset
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], confidences[i], class_ids[i]) for i in range(len(boxes)) if i in indexes]

# Function to determine lane position
def get_lane_position(x, width, frame_width):
    lane_width = frame_width / 3
    if x < lane_width:
        return "left"
    elif x < 2 * lane_width:
        return "center"
    else:
        return "right"

# Function to draw lane lines
def draw_lane_lines(frame):
    height, width, _ = frame.shape
    lane_width = width // 3
    cv2.line(frame, (lane_width, 0), (lane_width, height), (255, 0, 0), 2)
    cv2.line(frame, (2 * lane_width, 0), (2 * lane_width, height), (255, 0, 0), 2)

# Initialize variables for cut-in detection
previous_lanes = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Draw lane lines
    draw_lane_lines(frame)

    # Detect vehicles
    detections = detect_vehicles(frame)
    current_lanes = {}

    cut_in_detected = False

    for (box, confidence, class_id) in detections:
        x, y, w, h = box
        label = str(classes[class_id])
        lane_position = get_lane_position(x, width, width)

        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label + " " + lane_position, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        
        current_lanes[(x, y, w, h)] = lane_position

    # Check for cut-in
    for box, lane_position in current_lanes.items():
        if box in previous_lanes:
            prev_lane_position = previous_lanes[box]
            if prev_lane_position != lane_position:
                cv2.putText(frame, "Cut-In Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cut_in_detected = True
    
    previous_lanes = current_lanes

    if cut_in_detected:
        playsound('alarm.wav')

    cv2.imshow('Vehicle Cut-In Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
