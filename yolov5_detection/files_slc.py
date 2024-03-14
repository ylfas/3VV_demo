import os
import shutil

sels = ['train', 'val', 'test']
dir_path = '/home/zhyl_ylf/dataset/thumorprogram_data/malignant_tree/'

for sel in sels:
    dirs = os.path.join(dir_path, sel)
    dir_files = os.listdir(dirs)

    for dir in dir_files:
        files = os.listdir(os.path.join(dirs, dir))
        for file in files:
            if file.split('.')[-1] == 'jpg':
                shutil.copyfile(dirs + '/' + dir + '/' + file,
                                '/home/zhyl_ylf/dataset/thumorprogram_data/images/' + sel + '/' + 'malignant_' + dir + '_' + file)
            else:
                shutil.copyfile(dirs + '/' + dir + '/' + file,
                                '/home/zhyl_ylf/dataset/thumorprogram_data/annotations/' + sel + '/' + 'malignant_' + dir + '_' + file)


