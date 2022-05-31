import os, subprocess
import mouse
from face_control import Face
import cv2
import numpy as np

open_osk = subprocess.Popen('osk.exe', shell=True)
face = Face()
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
W, H = 0, 0
cnt = 0
BLINK_THRESH = 0.2
STEP = 2
STEP_RATE = 0.005
counter = 0

# biến nhớ ear của 2 mắt.
left_ear_old = 1
right_ear_old = 1

# đếm cqlick.
left_click = 0
right_click = 0
both_click = 0

while True:
    # Nhận frame từ webcam
    _, frame = webcam.read()
    # Gửi frame mới tới Face để phân tích
    face.refesh(frame)
    # Visualize (hiện tại chưa có)
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

        # if ear < BLINK_THRESH:
        #     counter += 1
        # else:
        #     if counter >= DEEP_CLICK_FRAME:
        #         mouse.click('right')
        #     elif counter >= CLICK_FRAME:
        #         mouse.click('left')
        #     counter = 0
        
        # if left_ear_old < BLINK_THRESH and left_ear > BLINK_THRESH and not (right_ear_old < BLINK_THRESH and right_ear > BLINK_THRESH):
        if left_ear < BLINK_THRESH:
            left_click += 1
        #     if left_click == 3:
        #         mouse.click('left')
        #         left_click = 0
        #         right_click = 0
        # # if right_ear_old < BLINK_THRESH and right_ear > BLINK_THRESH and not (left_ear_old < BLINK_THRESH and left_ear > BLINK_THRESH):
        if right_ear < BLINK_THRESH:
            right_click += 1
        #     if right_click == 3:
        #         mouse.click('right')
        #         right_click = 0
        #         left_click = 0
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
        
        left_ear_old = left_ear
        right_ear_old = right_ear

        # show ear threshold và ear.
        cv2.putText(frame, "left click: {:.2f}".format(left_click), (400, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "right click: {:.2f}".format(right_click), (400, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "CNT={}, w = {:.2f}, h = {:.2f}".format(cnt,w,h), (30,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
        
        # if cnt <= 600:
        #     if cnt <= 300:
        #         cv2.putText(frame, "Close your eyes for 5s after {:.2f}s ".format(5 - cnt/60), (180,200),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
        
        
        
        cv2.imshow("demo", frame)
    else:
        cv2.imshow("demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
