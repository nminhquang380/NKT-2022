# -*- coding: utf-8 -*-

from pkg_resources import resource_filename

def pose_predictor_model_location():
    return resource_filename(__name__, "shape_predictor_68_face_landmarks.dat")