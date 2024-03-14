yolov5_detection为yolov5的ROI检测模型训练：代码来自https://github.com/ultralytics/yolov5； <br />

![76107801_3VV_m](https://github.com/ylfas/3VV_demo/assets/110209878/ffc4de9a-d17e-4245-9e89-1d07a2c5f2ae)  ![76107801_3VV_m](https://github.com/ylfas/3VV_demo/assets/110209878/6b738b15-b75a-41c9-aee2-092de8c109fe)




**一阶段：** <br />
1.在'.../yolov5_detection/data/ab.yaml 文件中进行检测数据集的路径设置 <br />

2.在'.../yolov5_detection/train.py 文件中的配置路径，epoch数根据任务进行设定 <br />

3.运行 python train.py 进行检测训练； <br />

4.训练完成后，在'.../yolov5_detection/detect.py' 中，在'--weights'中输入训练好的权重文件路径 <br />

5.在detect.py中，调整参数'--conf-thres'，'--iou-thres'的阈值以获取更全面的检测效果，本文均设置为了0.3 <br />

6.在detect.py中，'--project'设置预测图的保存路径 <br />

7.运行 python detect.py 完成测试集的ROI预测。 <br />

**二阶段：**
1.在'.../3VV_demo/utils/detec_ROI_trim.py'中配置路径: 其中'predict_txt_path'为yolov5检测结果的txt文件，'full_size_img_path'为预测结果的路径，即上述在'--project'中设置的预测图的保存路径 <br />

2.在detec_ROI_trim.py中，配置'full_size_img_path_all'为初始全尺寸数据集的路径，在该数据集的测试集上进行裁剪获取ROI提取的新测试集，并在最后设定裁剪坐标记录的txt文件储存路径（'.../3VV_demo/result/new_dataset/test/txt/'） <br />

3.在detec_ROI_trim.py中，设定裁剪图像以及标签保存的路径（'.../3VV_demo/result/new_dataset/test/img（mask）/'） <br />

4.接着，在'.../3VV_demo/utils/label_ROI_trim.py'中配置路径，包括初始全尺寸数据集的训练集和验证集路径，以生成第二阶段中分割网络的新训练集以及测试集； <br />

5.最终根据上述新生成的数据集路径，在 '.../3VV_demo/dataset.py'中的MyDataset_trim类上配置新数据集的路径 <br />

6.在'.../3VV_demo/main_2th_stage.py'中设置预测图像的保存路径（'.../3VV_demo/result/2th_stage_seg/deeplabv3/'）以及恢复图像尺寸所需要的txt路径设置（'.../3VV_demo/result/new_dataset/test/txt/'） <br />

7.运行 python main_2th_stage.py进行二阶段的分割结果预测 <br />

8.在'.../3VV_demo/trim_predict.py'中根据分割预测结果的路径（'.../3VV_demo/result/2th_stage_seg/deeplabv3/'）以及原全尺寸数据集路径（'.../data/test/mask/'）进行路径配置 <br />

9.运行 python trim_predict.py，完成3VV结果的最终评估。 <br />
