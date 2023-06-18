import cv2

def load_camera(num=0):
    webcam = cv2.VideoCapture(num)
    while True:
        state,frame = webcam.read()
        if not state:
            break
        cv2.imshow("video",frame)
        if cv2.waitKey(1) == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()