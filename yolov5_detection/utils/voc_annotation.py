# import os
# import shutil
#
# slices = ['train','val','test']
#
# datadir_path = r'/home/zhyl_ylf/project/yolo/yolov5-5.0/yolov5-5.0/data_divide/'
# for slice in slices:
#     num_dirs = 0
#     all_files = 0
#     num_files = 0
#     dataload_dir = os.path.join(datadir_path, slice)
#     imagedirs = os.listdir(dataload_dir)
#     for dir in imagedirs:
#         num_dirs += 1
#         images = os.listdir(os.path.join(dataload_dir,dir))
#         for image in images:
#             all_files += 1
#             if image.split('.')[-1] in ['bmp','jpg']:
#                 num_files += 1
#                 shutil.copyfile(dataload_dir+'/'+dir+'/'+image, '/home/zhyl_ylf/project/yolo/yolov5-5.0/yolov5-5.0/data/images/%s_%s_%s'%(slice,num_files,image))
#
#     print(num_dirs,all_files,num_files)


import os
def rm(path1):
    # 返回当前目录下的内容。文件或文件夹
    # print(path)
    fls = os.listdir(path1)
    if len(fls)==0:
        # print('当前文件夹为空')
        print(f"删除:{path1}")
        # os.rmdir(path1)
        return

    for p in fls:
        p2 = f'{path1}\\{p}'
        if os.path.isdir(p2):
            # print(f'进入{p2}')
            rm(p2)
            if os.path.exists(p2) and len(os.listdir(p2)) == 0: # 里面删除后这个可能就是空文件了
                print(f"删除:{p2}")
                os.rmdir(p2) #在这里执行删除


if __name__ == '__main__':
    rm('/home/zhyl_ylf/project/yolo/yolov5-5.0/yolov5-5.0/data_divide/val/')
    # os.system('pause') # 按任意键退出

