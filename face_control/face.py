from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import os
import dlib
import cv2

# class Eye(object):
#     """
#     Class này định nghĩa mắt của người sử dụng.
#     Nó chứa các điểm đánh dấu của mắt (eye-landmarks).
#     Có phương thức tính ra độ nhắm của mắt.
#     """
#     LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
#     RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

#     def __init__(self):
#         self.frame = None
#         self.side = None
#         self.landnarks = None


class Face(object):
    """
    Class này ghi nhận khuôn mặt người dùng.
    Nó chứa điểm (face-landmarks) trên khuôn mặt.
    Có phương thức xác định điểm, xác định nhắm mắt (blink-detect), xác định cử chỉ đầu (headpose)
    """

    def __init__(self):
        self.frame = None
        self.nec_points = None
        self.left_eye = None
        self.right_eye = None
        # self.calibration = Calibration()

        #_face_detect được sử dụng để xác định khuôn mặt
        self._face_detector = dlib.get_frontal_face_detector()

        #_predictor được sử dụng để xác định những điểm trên khuôn mặt cho trước
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    def _analyze(self):
        """Xác định khuôn mặt"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        """Xác định điểm trên khuôn mặt"""
        landmarks = self._predictor(frame, faces[0])
        landmarks = face_utils.shape_to_np(landmarks)

        self.left_eye = landmarks[42:48]
        self.right_eye = landmarks[36:42]
        self.nec_points = np.float32([landmarks[17], landmarks[21], landmarks[22], landmarks[26], 
                            landmarks[36], landmarks[39], landmarks[42], landmarks[45], 
                            landmarks[31], landmarks[35], landmarks[48], landmarks[54], 
                            landmarks[57], landmarks[8]])

    def refesh(self, frame):
        """Để xác định lại cho frame mới"""
        self.frame = frame
        self._analyze()

    def left_eye_aspect_ratio(self):
        eye = self.left_eye
        #Tính chiều rộng khuôn mắt
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        #Tính chiều dài khuôn mắt
        C = dist.euclidean(eye[0], eye[3])
        #Tính tỉ lệ nhắm của mắt
        ear = (A + B) / (2.0 * C)
        return ear

    def right_eye_aspect_ratio(self):
        eye = self.right_eye
        #Tính chiều rộng khuôn mắt
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        #Tính chiều dài khuôn mắt
        C = dist.euclidean(eye[0], eye[3])
        #Tính tỉ lệ nhắm của mắt
        ear = (A + B) / (2.0 * C)
        return ear

    def get_head_pose_vector(self):
        """Xác định vector chỉ hướng đầu"""
        image_pts = self.nec_points
        #Một số parameters cần thiết
        K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
            0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
            0.0, 0.0, 1.0]
        D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
        dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

        object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                [1.330353, 7.122144, 6.903745],
                                [-1.330353, 7.122144, 6.903745],
                                [-6.825897, 6.760612, 4.402142],
                                [5.311432, 5.485328, 3.987654],
                                [1.789930, 5.393625, 4.413414],
                                [-1.789930, 5.393625, 4.413414],
                                [-5.311432, 5.485328, 3.987654],
                                [2.005628, 1.409845, 6.165652],
                                [-2.005628, 1.409845, 6.165652],
                                [2.774015, -2.080775, 5.048531],
                                [-2.774015, -2.080775, 5.048531],
                                [0.000000, -3.116408, 6.097667],
                                [0.000000, -7.415691, 4.070434]])

        reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                [10.0, 10.0, -10.0],
                                [10.0, -10.0, -10.0],
                                [10.0, -10.0, 10.0],
                                [-10.0, 10.0, 10.0],
                                [-10.0, 10.0, -10.0],
                                [-10.0, -10.0, -10.0],
                                [-10.0, -10.0, 10.0]])

        _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                            dist_coeffs)
        
        reprojectdst = tuple(map(tuple, (reprojectdst.reshape(8, 2))))

        x_start = int((reprojectdst[1][0] + reprojectdst[5][0] +reprojectdst[2][0] +reprojectdst[6][0]) / 4)
        y_start = int((reprojectdst[1][1] + reprojectdst[5][1] +reprojectdst[2][1] +reprojectdst[6][1]) / 4)
        start = (x_start, y_start)
        x_end = int((reprojectdst[0][0] + reprojectdst[4][0] +reprojectdst[3][0] +reprojectdst[7][0]) / 4)
        y_end = int((reprojectdst[0][1] + reprojectdst[4][1] +reprojectdst[3][1] +reprojectdst[7][1]) / 4)
        end = (x_end, y_end)
        return (start, end)


    """
    CẦN THÊM MỘT FUNCTION VISUALIZE
    """
    def annotated_frame(self):
        """Returns the main frame with pupils highlighted and draw contours for eyes"""
        frame = self.frame.copy()

        #draw contours for eyes

        cv2.drawContours(frame, [self.left_eye], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [self.right_eye], -1, (0, 255, 0), 1)

        return frame
    
        
    