import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math
import cv2 as cv

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 640

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def resize(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img


def annotated_img(image, result):
    # Copy image and parse result
    annotated_image = image.numpy_view()
    gestures, multi_hand_landmarks = result

    # Iterate through all landmarks
    for hand_landmarks in multi_hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])

        # Draw landmarks on image
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    return annotated_image
