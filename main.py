import os, subprocess
import mouse
from face_control import Face
import cv2
import numpy as np
import matplotlib.pyplot as plt

face = Face()
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
W, H = 0, 0
cnt = 0
BLINK_THRESH = 0.2
MOUTH_THRESH = 0.8
STEP = 2
STEP_RATE = 0.005
counter = 0

# đếm click
left_click = 0
right_click = 0
both_click = 0
mouth_click = 0

# biến đếm thời gian t
t = 0

while True:
    # Nhận frame từ webcam
    _, frame = webcam.read()
    # Gửi frame mới tới Face để phân tích
    face.refesh(frame)
    # Visualize 
    frame = face.annotated_frame()
    """Điều khiển chuột"""
    if face.check:
        (start, end) = face.get_head_pose_vector()
        vector = (end[0]-start[0], end[1]-start[1])
        h = end[1]-start[1]
        w = end[0]-start[0]
    
        # Tính độ mở của mắt
        left_ear = face.left_eye_aspect_ratio()
        right_ear = face.right_eye_aspect_ratio()
        ear = (left_ear+right_ear)/2

        # Tính độ nhướn của lông mày
        left_er = face.left_eyebrow_ratio()
        right_er = face.left_eyebrow_ratio()

        # Tính độ mở khuôn miệng
        mar = face.mouth_aspect_ratio()

        # Tinh chỉnh tham số W, H, đồng thời tính thời gian mắt nhắm
        if cnt <= 120:
            cnt += 1
            if abs(H) < 10 or abs(W) < 20:
                # W = (W*(cnt) + w)/(cnt)
                # H = (H*(cnt) + h)/(cnt)

                sign_w = -1 if (w-W) < 0 else 1
                W = W + sign_w*np.sqrt(abs(w - W))
                sign_h = -1 if (h-H) < 0 else 1
                H = H + sign_h*np.sqrt(abs(h - H)) 
                # BLINK_THRESH = (BLINK_THRESH*(cnt-301) + ear+0.1)/(cnt-300)
            

        
        """
        TODO:
        - Chưa tối ưu:
        - Khi initial vector quá lệch khỏi tâm thì việc điều khiển sẽ khó:
         + Option 1: Setup initial vector cố định (fixed cứng ở chính giữa)
         + Option 2: Intial chỉ được trong một khoảng cố định, dù tunning cũng không được vượt quá khoảng đó.
         + Option 3: Chỉ thay đổi công thức 
         
        """
        # Hành động
        if ((w/4)**2 + h**2) <= 1225:
            pass
        elif h-0.25*w >= 0 and h+0.25*w >= 0:
            mouse.move(0, STEP+int(((abs(h-H))**1.5)*STEP_RATE), absolute=False)
        elif h-0.25*w <= 0 and h+0.25*w <= 0:
            mouse.move(0, -(STEP+int(((abs(h-H))**1.5)*STEP_RATE)), absolute=False)
        elif h-0.25*w > 0 and h+0.25*w < 0:
            mouse.move(+STEP+int(((abs(w-W))**1.5)*0.5*STEP_RATE), 0, absolute=False)
        elif h-0.25*w < 0 and h+0.25*w > 0:
            mouse.move(-STEP-int(((abs(w-W))**1.5)*0.5*STEP_RATE), 0, absolute=False)

        # điều khiển click:
        # TODO: right-click, left-click, middle-click.

        if left_ear < BLINK_THRESH:
            left_click += 1
        
        if right_ear < BLINK_THRESH:
            right_click += 1
       
        if left_click >= 2 and right_click < 2:
            mouse.click('left')
            left_click = 0
            right_click = 0
        elif left_click < 2 and right_click >= 2:
            mouse.click('right')
            left_click = 0
            right_click = 0
        elif left_click > 2 and right_click > 2:
            mouse.click('middle')
            left_click = 0
            right_click = 0

        if left_ear >= BLINK_THRESH:
            left_click = 0
        if right_ear >= BLINK_THRESH:
            right_click = 0
        
        # Popup virtual keyboard
        if mar >= MOUTH_THRESH:
            mouth_click += 1
        else:
            if mouth_click >= 30:
                open_osk = subprocess.Popen('osk.exe', shell=True)
            mouth_click = 0
        
        

        # cv2.putText(frame, "left click: {:.2f}".format(left_click), (400, 30),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # cv2.putText(frame, "right click: {:.2f}".format(right_click), (400, 60),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "ler = {:.2f}, rer = {:.2f}".format(left_er, right_er), (30,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "mar = {:.2f}".format(mar), (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("demo", frame)

    else:
        cv2.imshow("demo", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()
