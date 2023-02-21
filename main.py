import cv2
from cv2 import VideoCapture, CascadeClassifier, Mat
import os

def main() -> None:
    """Main function. Executes the face detection on the webcam function.
    """
    face_detector_path = os.path.join(os.getcwd(), "Assets", "Cascades","haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(face_detector_path)
    
    camera = cv2.VideoCapture(0)

    detect_faces_on_camera(camera, face_detector)

    camera.release()
    cv2.destroyAllWindows()


def detect_faces_on_camera(camera:VideoCapture, face_detector:CascadeClassifier) -> None:
    """Detects faces on camera and draws a rectangle around them. Quits when 'q' is pressed.
    
    Args:
        camera (VideoCapture): Webcam object.
        face_detector (CascadeClassifier): Face detector object.
    """
    # Capture frame-by-frame
    while True:
        ret, frame = camera.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = face_detector.detectMultiScale(gray_frame, minSize=(100, 100),
                                                    minNeighbors=5)
        
        draw_rectangle_around_faces(frame, detections)

        cv2.imshow('Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def draw_rectangle_around_faces(frame:Mat, faces:list[tuple]) -> None:
    """Draws a rectangle around the faces.

    Args:
        frame (Mat): Frame to draw on.
        faces (list[tuple]): List of faces.
    """
    for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

if __name__ == "__main__":
    main()
