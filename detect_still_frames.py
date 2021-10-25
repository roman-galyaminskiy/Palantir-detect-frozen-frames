import numpy as np
import cv2

# Path to video file
FILE_PATH = "IMG_8914.MOV"
cap = cv2.VideoCapture(FILE_PATH)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if not cap.isOpened():
    print('Unable to open file')
    exit(0)

# Constants and thresholds
DIFF_THRESHOLD = 0.001
DETECTION_THRESHOLD_MILLIS = 500
DEBOUNCE_THRESHOLD_MILLIS = 100
SCALING_FACTOR = 0.5
MAX_FRAME_SIZE = frame_width * frame_height * SCALING_FACTOR ** 2 * 3 * 255
LINE_WIDTH = 10
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def frame_diff(prev_frame, cur_frame, next_frame):
    """
    Calculates absolute difference between pair of two consecutive frames. Returns the 
    result of bitwise 'AND' between the above two resultant images to obtain a mask where
    only the areas with white pixels are shown
    """
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
    return cv2.bitwise_and(diff_frames1, diff_frames2)

prev_frame, cur_frame, next_frame = None, None, None
cur_frame_time, prev_frame_time = None, None
state, prev_state, reading, prev_reading = False, False, False, False
still_frame_timer, last_debounce_time = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.resize(frame, None, fx=SCALING_FACTOR, fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA)

    # Get 3 frames to calc diff
    prev_frame = cur_frame
    cur_frame = next_frame
    next_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    prev_frame_time = cur_frame_time

    # Get current frame timestamp
    cur_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    
    # Calc diff if ready
    if prev_frame is not None:
        diff = frame_diff(prev_frame, cur_frame, next_frame)
        sum = np.sum(diff)    

        # If diff is lower than threshold, mark frame as frozen
        prev_state = state
        if (sum/MAX_FRAME_SIZE < DIFF_THRESHOLD):
            reading = True
        else:
            reading = False

        # Frame state debouncing
        still_frame_timer += cur_frame_time - prev_frame_time
        if (reading != prev_reading):
            last_debounce_time = cur_frame_time

        if ((cur_frame_time - last_debounce_time) > DEBOUNCE_THRESHOLD_MILLIS):
            if (reading != state):
                state = reading
                still_frame_timer = 0

        prev_reading = reading

        if (state and still_frame_timer > DETECTION_THRESHOLD_MILLIS):            
            color = GREEN
        else:
            color = RED

        # Save frozen frames to disc
        if (not prev_state and state):
            cv2.imwrite(str(int(cur_frame_time)) + ".jpeg", frame)
        
        # Draw windows 
        cv2.rectangle(frame, (0, 0), (diff.shape[1], diff.shape[0]), color, 10, lineType=cv2.LINE_4)
        cv2.rectangle(frame, (10, 2), (150,50), (255,255,255), -1)
        cv2.putText(frame, "Millis: " + str(round(cur_frame_time, 1)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.putText(frame, "Diff: " + str(round(sum/MAX_FRAME_SIZE, 5)), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        cv2.imshow('frame', frame) 
        cv2.imshow('diff', diff)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()