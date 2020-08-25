from pathlib import Path

import cv2
import insightface
import numpy as np


model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=-1, nms=0.4) # prepare the enviorment; non-max surpression threshold is set to 0.4
model_face_detection_only = insightface.model_zoo.get_model('retinaface_r50_v1')
model_face_detection_only.prepare(ctx_id = -1, nms=0.4)

def detect_face_comprehensive(frame, model):
    for face in model.get(frame):
        x, y, x2, y2 = face.bbox.astype(np.int).flatten()
        cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 1)
        cv2.putText(frame, f'age: {face.age}', (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,0), thickness=1)
        cv2.putText(frame, 'female' if face.gender==0 else 'male', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,0), thickness=1)
    return frame

def detect_face(frame, model):
    bounding_boxes, landmark = model.detect(frame, threshold=0.5, scale=1.0)
    for bounding_box in bounding_boxes:
        x, y, x2, y2, _ = bounding_box.astype(np.int).flatten()
        cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 1)
    return frame

path_input = Path('./tests/videos/airport.mp4')
path_output = path_input.parent / (path_input.stem + '_detected' + path_input.suffix)
video_input = cv2.VideoCapture(str(path_input))
fps = int(video_input.get(cv2.CAP_PROP_FPS))
width, height = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_output = cv2.VideoWriter(str(path_output), int(video_input.get(cv2.CAP_PROP_FOURCC)), fps, (width, height))

while video_input.isOpened():
    ret, frame_input = video_input.read()
    if ret==True: # check whether frame is read correctly or reaching the end of video
        # frame_output = detect_face_comprehensive(frame_input, model)
        frame_output = detect_face(frame_input, model_face_detection_only)
        video_output.write(frame_output)
        cv2.imshow('face detection demo', frame_output)
        if cv2.waitKey(1) & 0xFF == ord('q'): # press 'q' to exit
            break
    else:
        break

video_input.release()
video_output.release()
cv2.destroyAllWindows()