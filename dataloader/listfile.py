import os
import os.path

IMG_EXTENSIONS = ['.PNG', '.png']
PARA_EXTENSIONS = ['.JSON', '.json']

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_json_file(filename):
    return any(filename.endswith(extension) for extension in PARA_EXTENSIONS)

def get_keyframe(dataset):
    keyframe_path = []
    for sub in dataset:
        keyframes = os.listdir(sub)
        for skf in keyframes:
            if 'ignore' not in skf:
                keyframe_path.append(os.path.join(sub,skf))

    return keyframe_path

def get_data(keyframe_sets, test=False):
    """
    This function will get the data path for training
    Args: 
    keyframe_sets: list of paths to keyframe folders
    test: indicator to tell if the the folder test set or other kinds.
    """
    left_img_sets = []
    right_img_sets = []
    camera_para_sets = []
    for kf in keyframe_sets:
        left_img_path = kf + '/left_finalpass/'
        right_img_path = kf + '/right_finalpass/'
        
        if not test:
            camera_para_path = kf + '/frame_data/'
            for para in os.listdir(camera_para_path):
                para_file = camera_para_path + para

                if is_json_file(para_file):
                    left_img_file = left_img_path + para.split('.')[0] + '.png'
                    right_img_file = right_img_path + para.split('.')[0] + '.png'
                    camera_para_sets.append(para_file)
                    left_img_sets.append(left_img_file)
                    right_img_sets.append(right_img_file)


        if test:
            para_file = kf+'/endoscope_calibration.yaml'
            camera_para_sets.append(para_file)

            for img_file in os.listdir(left_img_path):
                left_img_file =  left_img_path + img_file
                right_img_file = right_img_path + img_file
                if is_img_file(left_img_file):
                    left_img_sets.append(left_img_file)
                if is_img_file(right_img_file):
                    right_img_sets.append(right_img_file)

        
    return left_img_sets, right_img_sets, camera_para_sets



def SCARED_lister(filepath):
    train_path = filepath + '/train'
    val_path = filepath + '/val'
    test_path = filepath + '/test'


    # this part get all the datasets path (dataset1, dataset2, dataset3)
    train_sets = [os.path.join(train_path,s) for s in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, s))]
    val_sets = [os.path.join(val_path,s) for s in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, s))]
    test_sets = [os.path.join(test_path,s) for s in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, s))]

    train_keyframes = get_keyframe(train_sets)
    val_keyframes = get_keyframe(val_sets)
    test_keyframes = get_keyframe(test_sets)

    # for train and validataion sets, 'get_data' returns 
    train_left_img, train_right_img, train_cam_para = get_data(train_keyframes)
    val_left_img, val_right_img, val_cam_para = get_data(val_keyframes)
    test_left_img, test_right_img, test_cam_para = get_data(test_keyframes, test=True)

    return train_left_img[:1000], train_right_img[:1000], train_cam_para[:1000], val_left_img, val_right_img, val_cam_para, test_left_img, test_right_img, test_cam_para


# def test():
#     filepath = '/media/xiran_zhang/2011_HDD7/EndoVis_SCARED'
#     SCARED_lister(filepath)


# if __name__ == '__main__':
#     test()