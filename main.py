import mouse
from face_control import Face
import cv2

face = Face()
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
W, H = 140, 35
BLINK_THRESH = 0.15
CLICK_FRAME = 3
DEEP_CLICK_FRAME = 9
counter = 0
while True:
    # Nhận frame từ webcam
    _, frame = webcam.read()
    # Gửi frame mới tới Face để phân tích
    face.refesh(frame)
    # Visualize (hiện tại chưa có)
    frame = face.annotated_frame()
    """Điều khiển chuột"""
    (start, end) = face.get_head_pose_vector()
    vector = (end[0]-start[0], end[1]-start[1])
    h = end[1]-start[1]
    w = end[0]-start[0]
    left_ear = face.left_eye_aspect_ratio()
    right_ear = face.right_eye_aspect_ratio()
    ear = (left_ear+right_ear)/2

    # Hành động
    if ((w/4)**2 + h**2) <= 1225:
        pass
    
    elif h-0.25*w >= 0 and h+0.25*w >= 0:
        mouse.move(0, int(h-H))
    elif h-0.25*w <= 0 and h+0.25*w <= 0:
        mouse.move(0, -int(h-H))
    elif h-0.25*w > 0 and h+0.25*w < 0:
        mouse.move(int(w-W), 0)
    elif h-0.25*w < 0 and h+0.25*w > 0:
        mouse.move(-int(w-W), 0)
    # else:
    #     mouse.move(int(-(w-W)), int((h-H)), absolute=False)
    if ear < BLINK_THRESH:
        counter += 1
    else:
        if counter >= DEEP_CLICK_FRAME:
            mouse.click('right')
        elif counter >= CLICK_FRAME:
            mouse.click('left')
        counter = 0

    cv2.putText(frame, "left EAR: {:.2f}".format(ear), (500, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
