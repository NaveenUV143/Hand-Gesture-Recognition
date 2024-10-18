import cv2
import mediapipe as mp
import numpy as np

class HandGestureRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, 
                                          max_num_hands=1, 
                                          min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_hand(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, 
                                                self.mp_hands.HAND_CONNECTIONS)
            return results.multi_hand_landmarks[0]
        return None

    def recognize_gesture(self, landmarks):
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        if thumb_tip.y < index_tip.y and middle_tip.y > index_tip.y and ring_tip.y > index_tip.y and pinky_tip.y > index_tip.y:
            return "1"
        if index_tip.y < middle_tip.y and ring_tip.y > middle_tip.y and pinky_tip.y > middle_tip.y:
            return "2"
        if index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and pinky_tip.y > ring_tip.y:
            return "3"
        if index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and ring_tip.y < pinky_tip.y:
            return "4"
        if index_tip.y < middle_tip.y and middle_tip.y < ring_tip.y and ring_tip.y < pinky_tip.y and thumb_tip.y < index_tip.y:
            return "5"
        
        if (thumb_tip.y < index_tip.y and 
            thumb_tip.y < middle_tip.y and 
            thumb_tip.y < ring_tip.y and 
            thumb_tip.y < pinky_tip.y):
            return "Thumbs Up"

        if (index_tip.y < middle_tip.y and 
            middle_tip.y < ring_tip.y and 
            ring_tip.y < pinky_tip.y and 
            index_tip.x < middle_tip.x):
            return "Peace"

        if (index_tip.y > thumb_tip.y and 
            middle_tip.y > thumb_tip.y and 
            ring_tip.y > thumb_tip.y and 
            pinky_tip.y > thumb_tip.y):
            return "Fist"

        if (index_tip.y < thumb_tip.y and 
            middle_tip.y < thumb_tip.y and 
            ring_tip.y < thumb_tip.y and 
            pinky_tip.y < thumb_tip.y):
            return "Open Hand"

        return "Unknown Gesture"
