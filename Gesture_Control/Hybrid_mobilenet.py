import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import pyautogui

# Load your trained model
model = load_model(r"C:\Users\rahul\Desktop\archive\Hybrid_mobilenetv2_model_finetuned.h5")

# Initialize MediaPipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the primary camera

# Dictionary to map model output to gestures/classes
class_labels = {
    0: 'volume_up', 1: 'volume_down', 2: 'C', 3: 'Pause', 4: 'Fast backward',11: 'Fast forward', 22: 'Window_close'
}

# Model prediction threshold
prediction_threshold = 0.4  # Experiment with different threshold values
 
# Main loop
while True:
    ret, frame = cap.read()

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Flip the frame horizontally for correct handedness output
    frame_flipped = cv2.flip(frame_rgb, 1)

    # Detect hands in the frame
    results = hands.process(frame_flipped)

    # If hands detected, proceed with gesture prediction
    if results.multi_hand_landmarks:
        # Resize the frame to match the input size of the model (224x224)
        resized_frame = cv2.resize(frame, (224, 224))

        # Normalize the frame
        normalized_frame = resized_frame / 255.0

        # Reshape the frame to match the input shape of the model
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Use the model to predict the gesture
        prediction = model.predict(input_frame)

        # Model prediction threshold
        if np.max(prediction) > prediction_threshold:
            gesture_id = np.argmax(prediction)
            predicted_class = class_labels.get(gesture_id, 'unknown')
            # Media player control actions
            if predicted_class == 'volume_up':
                pyautogui.press('volumeup')
            elif predicted_class == 'volume_down':
                pyautogui.press('volumedown')
            elif predicted_class in ['Play', 'Pause']:
                pyautogui.press('playpause')
            elif predicted_class == 'Fast forward':
                pyautogui.press('right')
            elif predicted_class == 'Fast backward':
                pyautogui.hotkey('left')
            elif predicted_class == 'Window_close':
                pyautogui.sleep(0.5)
                pyautogui.hotkey('alt', 'f4')
        else:
            predicted_class = 'unknown'
    else:
        predicted_class = 'No hand detected'

    # Display the predicted class on the video window
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video window
    cv2.imshow('Hand Gesture Recognition', frame)

    # Press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
