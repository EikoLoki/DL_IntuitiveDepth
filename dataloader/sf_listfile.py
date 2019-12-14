import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    classes = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
    image = [c + '/frames_finalpass' for c in classes]
    disp  = [c + '/disparity' for c in classes]
    # print(classes)
    # print(image)
    # print(disp)

    all_left_img=[]
    all_right_img=[]
    all_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    monkaa_empty = True
    flying_empty = True
    driving_empty = True

    for c in classes:
        if 'Monkaa' in c:
            monkaa_empty = False
        if 'Flying' in c:
            flying_empty = False
        if 'Driving' in c:
            driving_empty = False


    # ========================= flying =======================
    if not flying_empty:

        flying_path = filepath + [x for x in image if 'Flying' in x][0]
        flying_disp = filepath + [x for x in disp if 'Flying' in x][0]
        flying_dir = flying_path+'/TRAIN/'
        subdir = ['A','B','C']

        for ss in subdir:
            flying = os.listdir(flying_dir+ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
                for im in imm_l:
                    if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                        all_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

                    all_left_disp.append(flying_disp+'/TRAIN/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                    if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                        all_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

        flying_dir = flying_path+'/TEST/'

        subdir = ['A','B','C']

        for ss in subdir:
            flying = os.listdir(flying_dir+ss)

            for ff in flying:
                imm_l = os.listdir(flying_dir+ss+'/'+ff+'/left/')
                for im in imm_l:
                    if is_image_file(flying_dir+ss+'/'+ff+'/left/'+im):
                        test_left_img.append(flying_dir+ss+'/'+ff+'/left/'+im)

                    test_left_disp.append(flying_disp+'/TEST/'+ss+'/'+ff+'/left/'+im.split(".")[0]+'.pfm')

                    if is_image_file(flying_dir+ss+'/'+ff+'/right/'+im):
                        test_right_img.append(flying_dir+ss+'/'+ff+'/right/'+im)

    return all_left_img[:2000], all_right_img[:2000], test_left_img[:1000], test_right_img[:1000]


