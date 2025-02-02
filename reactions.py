from utils import *
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Before running this code, please install mediapipe:
# pip install -q mediapipe==0.10.0

# If there are errors with your protobuf version (builder.py), follow these steps to fix:
# https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal

global can_inference, transforms
DEBUG = False


# *** FUNCTIONS ***
def reset_videos():
    rain.set(cv.CAP_PROP_POS_FRAMES, 0)
    thumb_up.set(cv.CAP_PROP_POS_FRAMES, 0)
    confetti.set(cv.CAP_PROP_POS_FRAMES, 0)
    balloons.set(cv.CAP_PROP_POS_FRAMES, 0)


def check_gestures(num_gesture, top_gesture):
    global can_inference, transforms
    cat1 = top_gesture[0][0].category_name
    # Check and set corresponding transforms to true
    if num_gesture == 1:
        if cat1 == 'Thumb_Up':
            print('Thumb_Up')
            transforms['Thumb_Up'] = True
            can_inference = False
        elif cat1 == 'Victory':
            print('Victory')
            transforms['Victory'] = True
            can_inference = False
        elif cat1 == 'Open_Palm':
            print('Open_Palm')
            transforms['Open_Palm'] = True
            can_inference = False
    if num_gesture == 2:
        cat2 = top_gesture[1][0].category_name
        if cat1 == 'Thumb_Down' and cat2 == 'Thumb_Down':
            print('Two_Thumb_Down')
            transforms['Two_Thumb_Down'] = True
            can_inference = False
        elif cat1 == 'Victory' and cat2 == 'Victory':
            print('Two_Victory')
            transforms['Two_Victory'] = True
            can_inference = False
        elif cat1 == 'Thumb_Up' or cat2 == 'Thumb_Up':
            print('Thumb_Up')
            transforms['Thumb_Up'] = True
            can_inference = False
        elif cat1 == 'Open_Palm' or cat2 == 'Open_Palm':
            print('Open_Palm')
            transforms['Open_Palm'] = True
            can_inference = False
        elif cat1 == 'Victory' or cat2 == 'Victory':
            print('Victory')
            transforms['Victory'] = True
            can_inference = False


def apply_rain(frame):
    # Read the overlay frame
    ret_overlay, overlay_frame = rain.read()

    if not ret_overlay:
        # Rewind video if ended
        rain.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, overlay_frame = rain.read()

    # Resize to frame
    overlay_frame = cv.resize(overlay_frame, (frame.shape[1], frame.shape[0]))

    # Blend the webcam and overlay
    alpha = 0.5  # Adjust the alpha value to control the blending intensity
    frame = cv.addWeighted(frame, 1 - alpha, overlay_frame, alpha, 0)

    return frame


def apply_thumb_up(frame):
    overlay_x, overlay_y = 400, 50  # Position
    overlay_width, overlay_height = int(960 / 5), int(930 / 5)  # Size

    # Read the overlay frame
    ret_overlay, overlay_frame = thumb_up.read()

    if not ret_overlay:
        # Rewind video if ended
        thumb_up.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, overlay_frame = thumb_up.read()

    # Resize overlay to region of interest
    overlay_frame = cv.resize(overlay_frame, (overlay_width, overlay_height))

    # Remove green screen
    _, alpha_mask = cv.threshold(overlay_frame[:, :, 0], 200, 255, cv.THRESH_BINARY)
    alpha_mask_inv = cv.bitwise_not(alpha_mask)

    # Extract roi for overlay and webcam, then overlay
    roi = frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width]
    bg = cv.bitwise_and(roi, roi, mask=alpha_mask_inv)
    overlay_content = cv.bitwise_and(overlay_frame, overlay_frame, mask=alpha_mask)
    blended_roi = cv.addWeighted(bg, 1, overlay_content, 1, 0)
    frame[overlay_y:overlay_y + overlay_height, overlay_x:overlay_x + overlay_width] = blended_roi

    return frame


def apply_confetti(frame):
    # Read the overlay frame
    ret_overlay, overlay_frame = confetti.read()

    if not ret_overlay:
        # Rewind video if ended
        confetti.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, overlay_frame = confetti.read()

    # Crop the confetti frame at the center
    overlay_height, overlay_width, _ = overlay_frame.shape
    crop_start_x = max((overlay_width - frame.shape[1]) // 2, 0)
    crop_start_y = max((overlay_height - frame.shape[0]) // 2, 0)
    crop_end_x = min(crop_start_x + frame.shape[1], overlay_width)
    crop_end_y = min(crop_start_y + frame.shape[0], overlay_height)
    overlay_frame = overlay_frame[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

    # Remove green screen
    _, alpha_mask = cv.threshold(overlay_frame[:, :, 0], 200, 255, cv.THRESH_BINARY)
    alpha_mask_inv = cv.bitwise_not(alpha_mask)

    # Extract background from webcam using bitwise
    bg = cv.bitwise_and(frame, frame, mask=alpha_mask_inv)

    # Overlay and blend
    overlay_content = cv.bitwise_and(overlay_frame, overlay_frame, mask=alpha_mask)
    frame = cv.addWeighted(bg, 1, overlay_content, 1, 0)

    return frame


def apply_balloons(frame):
    # Read the overlay frame
    ret_overlay, overlay_frame = balloons.read()

    if not ret_overlay:
        # Rewind video if ended
        balloons.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, overlay_frame = balloons.read()

    # Resize to frame
    overlay_frame = cv.resize(overlay_frame, (frame.shape[1], frame.shape[0]))

    # Remove green screen
    _, alpha_mask = cv.threshold(overlay_frame[:, :, 2], 65, 255, cv.THRESH_BINARY)
    alpha_mask_inv = cv.bitwise_not(alpha_mask)

    # Extract background from webcam using bitwise
    bg = cv.bitwise_and(frame, frame, mask=alpha_mask_inv)

    # Overlay and blend
    overlay_content = cv.bitwise_and(overlay_frame, overlay_frame, mask=alpha_mask)
    frame = cv.addWeighted(bg, 1, overlay_content, 1, 0)

    return frame


def apply_space(frame):
    # Read the overlay frame
    ret_overlay, overlay_frame = space.read()

    if not ret_overlay:
        # Rewind video if ended
        space.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, overlay_frame = space.read()

    # Convert frame to mp Image object
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Segment image and return mask
    segmentation_result = segmenter.segment(mp_frame)
    category_mask = segmentation_result.category_mask

    # Convert and resize tp frame
    image_data = cv.cvtColor(mp_frame.numpy_view(), cv.COLOR_BGR2RGB)
    background_frame = cv.resize(overlay_frame, (frame.shape[1], frame.shape[0]))

    # Apply background
    condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.1
    frame = np.where(condition, image_data, background_frame)

    return frame


# *** REST OF CODE ***

# Read in videos for effects:
rain = cv.VideoCapture('videos/rain.mp4')
thumb_up = cv.VideoCapture('videos/thumbs_up.mp4')
confetti = cv.VideoCapture('videos/confetti.mp4')
balloons = cv.VideoCapture('videos/balloons.mp4')
space = cv.VideoCapture('videos/space.mp4')
fps = space.get(cv.CAP_PROP_FPS)

# Create a GestureRecognizer object
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Create a Segmenter object
base_options_seg = python.BaseOptions(model_asset_path='deeplab_v3.tflite')
options_seg = vision.ImageSegmenterOptions(base_options=base_options_seg, output_category_mask=True)
segmenter = vision.ImageSegmenter.create_from_options(options_seg)


# Main webcam loop
def run_webcam():
    global can_inference, transforms

    # Setup Cam:
    cv.namedWindow("Camera Stream")
    cap = cv.VideoCapture(0)

    # Transformation Bools
    transforms = {'Thumb_Up': False, 'Thumb_Down': False, 'Victory': False, 'Two_Thumb_Up': False,
                  'Two_Thumb_Down': False, 'Two_Victory': False, 'Open_Palm': False}

    # Implement buffer for effects
    buffer = 5.0
    activation_time = 0.0
    can_inference = True

    # Loop for video feed
    while True:
        # Read frame
        ret, frame = cap.read()
        key = cv.waitKey(1)

        # If buffer time has passed, can inference again, reset all transforms and videos
        current_time = time.time()
        if current_time - activation_time > buffer:
            can_inference = True
            for transform_name in transforms:
                transforms[transform_name] = False
            reset_videos()  # Reset video frames

        # Load and resize frame as mp image
        mp_frame = frame[:, :, ::-1]
        mp_frame = resize(mp_frame)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_frame)

        # Perform Inference and process
        recognition_result = recognizer.recognize(mp_frame)

        # Checks if hand is recognized
        if len(recognition_result.gestures) > 0:
            # Parse result
            num_gesture = len(recognition_result.gestures)
            top_gesture = recognition_result.gestures
            hand_landmarks = recognition_result.hand_landmarks
            result = (top_gesture, hand_landmarks)

            # Checks if gesture is detected for 1 and 2 hands
            bool_statement = False
            if num_gesture == 1:
                bool_statement = top_gesture[0][0].category_name != 'None'
            elif num_gesture == 2:
                bool_statement = (top_gesture[0][0] != 'None' and top_gesture[1][0] != 'None')

            if bool_statement and can_inference:
                # Run check gesture function to evaluate each gesture
                check_gestures(num_gesture, top_gesture)

                # Update timestamp
                activation_time = time.time()

                # Annotate frame
                if DEBUG:
                    frame = annotated_img(mp_frame, result)
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    cv.putText(frame, top_gesture[0][0].category_name, (100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (0, 0, 255), 1)
                    if num_gesture == 2:
                        cv.putText(frame, top_gesture[1][0].category_name, (400, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                                   (0, 0, 255), 1)
                    txt = "num gestures: " + str(num_gesture)
                    cv.putText(frame, txt, (100, 300), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    frame = mp_frame.numpy_view()
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Show buffer time
        if DEBUG:
            buf_text = str(round(buffer - (current_time - activation_time), 2))
            cv.putText(frame, 'buffer_time: ' + buf_text, (100, 400), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # If transform is true, apply effects to frame:
        if transforms['Two_Thumb_Down']:
            frame = apply_rain(frame)
        if transforms['Thumb_Up']:
            frame = apply_thumb_up(frame)
        if transforms['Two_Victory']:
            frame = apply_confetti(frame)
        if transforms['Victory']:
            frame = apply_balloons(frame)
        if transforms['Open_Palm']:
            frame = apply_space(frame)

        # Show frame
        cv.imshow("Camera Stream", frame)

        # Use ESC to exit
        if key == 27:
            break

    # CLose feed
    cap.release()
    cv.destroyAllWindows()


def main():
    run_webcam()


if __name__ == "__main__":
    main()
