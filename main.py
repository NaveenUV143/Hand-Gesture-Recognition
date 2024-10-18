import cv2
from hand_gesture import HandGestureRecognition

def main():
    cap = cv2.VideoCapture(0)
    gesture_recognition = HandGestureRecognition()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        landmarks = gesture_recognition.detect_hand(frame)
        
        if landmarks:
            gesture = gesture_recognition.recognize_gesture(landmarks)
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
