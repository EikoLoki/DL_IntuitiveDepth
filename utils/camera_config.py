import os
import os.path 
import numpy as np
import json

def read_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data['camera-calibration']


def get_camera_intrin(data):
    l_camera_intrin = np.asarray(data['KL'])
    r_camera_intrin = np.asarray(data['KR'])
    return l_camera_intrin, r_camera_intrin


def get_camera_pose(data):
    rotation = np.asarray(data['R'])
    translation = np.asarray(data['T'])
    return rotation, translation

def get_camera_dist(data):
    l_camera_dist = np.asarray(data['DL'])
    r_camera_dist = np.asarray(data['DR'])
    return l_camera_dist, r_camera_dist
    

class CameraPara:
    def __init__(self, file_path):
        self.l_camera_intrin = np.zeros((3,3))
        self.r_camera_intrin = np.zeros((3,3))
        self.l_dist_coeff = np.zeros((5,1))
        self.r_dist_coeff = np.zeros((5,1))
        self.rotation = np.eye(3)
        self.translation = np.zeros((3,1))

        data = read_json(file_path)
        self.l_camera_intrin, self.r_camera_intrin = get_camera_intrin(data)
        self.l_dist_coeff, self.r_camera_dist = get_camera_dist(data)
        self.rotation, self.translation = get_camera_pose(data)



# unit test
def test_CameraPara():
    endopara = CameraPara('/media/xiran_zhang/2011_HDD7/EndoVis_SCARED/train/dataset1/keyframe_2/frame_data/frame_data000000.json')
    print('Camera Intrinsic:', endopara.l_camera_intrin, endopara.r_camera_intrin)
    print('Camera distortion:', endopara.l_dist_coeff, endopara.r_dist_coeff)
    print('Camera pose:', endopara.rotation, endopara.translation)


if __name__ == '__main__':
    test_CameraPara()