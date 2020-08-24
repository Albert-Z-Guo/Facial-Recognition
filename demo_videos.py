import cv2
import insightface
import numpy as np

model = insightface.model_zoo.get_model('retinaface_r50_v1')
model.prepare(ctx_id = -1, nms=0.4)

def detect_face(frame):
    bbox, landmark = model.detect(frame, threshold=0.5, scale=1.0)
    for face_bbox in bbox:
        x, y, x2, y2, _ = face_bbox.astype(np.int).flatten()
        cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 1)
    return frame

video = cv2.VideoCapture('/Users/guo/Documents/GitHub/Facial Recognition/tests/videos/airport.mp4')
while(True):
    ret, frame = video.read()
    cv2.imshow('facial detection', detect_face(frame))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()