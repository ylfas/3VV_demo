# **A Deep Learning Framework for Identifying and Segmenting Three Vessels in Fetal Heart Ultrasound Images**

![image](https://github.com/ylfas/3VV_demo/assets/110209878/89eb2dce-78ab-4114-8c0e-2fb1d459719d)

* In this study, we propose a deep learning-based framework for the identification and segmentation of the three vessels — the pulmonary artery, aorta, and superior vena cava — in the ultrasound three vessel view (3VV) of the fetal heart.  In the first stage of the framework, the object detection model Yolov5 is employed to identify the three vessels and localize the Region of Interest (ROI) within the original full-sized ultrasound images.  Subsequently, a modified Deeplabv3 equipped with our novel AMFF (Attentional Multi-scale Feature Fusion) module is applied in the second stage to segment the three vessels within the cropped ROI images.


# **The framework's flowchart**
![image](https://github.com/ylfas/3VV_demo/assets/110209878/86b7bdaf-624c-4fb7-b1ca-c411e69f7d73) <br />

* We design a deep-learning based framework for the accurate segmentation of the three vessels within the three-vessel plane of the fetal heart, namely, the aorta, the pulmonary artery and the superior vena cava. <br />
* In the first stage of the network framework, a detection model is employed to extract the Regions of Interest (ROIs) containing important regions in the image and perform cropping.
* In the second stage of the network framework, an improved version of Deeplabv3 equipped with a multi-scale feature fusion module is utilized for fine segmentation of multiple categories of blood vessel outlines. <br />

****

# **AMFF(Attentional Multi-scale Feature Fusion module)**

![image](https://github.com/ylfas/3VV_demo/assets/110209878/ce30518f-09c4-472e-b037-efa72af883d1) <br />
* Accroding to the structure of AMFF module.  It consists of multiple feature extraction branches with convolutions of different dilation rates to obtain features with diverse receptive fields.  To ensure that each branch preserves small object features, we encourage interaction among branches by integrating features through hierarchical connections.  Furthermore, we introduce spatial attention operations to selectively enhance the most effective features of each branch, thereby improving feature representations at multiple scales.  Subsequently, the features from all branches are concatenated to create fused features that retain information related to multi-scale targets.  The fused feature (2048x32x32) is then dimensionally reduced to three channels through two convolutional layers, with each channel predicting one type of vessel.  Finally, the prediction is upsampled eight times through bilinear interpolation to restore it to the original image resolution.

# **Improved deeplabv3 model**
![image](https://github.com/ylfas/3VV_demo/assets/110209878/9b1a3e04-0306-4c80-bc49-0b527a39f7b2)  <br />

* The second stage of our framework is a modified Deeplabv3 equipped with our novel AMFF module for instance segmentation of the three vessels.   The AMMF’s architecture is illustrated.  A cascade of ResNet34 blocks are used to encode image features.  To be concrete, the initial phase involves an initialization block, which consists of a 7x7 convolution with a stride of 2, a padding of 3, and a Batch Normalization (BN) layer. 


****
# **Usage**

## **For directed application:**
If you want to use our pre-trained model on your own dataset, please follow the steps below in order. Before usage, you should put your dataset images in './dataset/test/'. Then, you should run the following two codes. And last, the prediction result will in './3VV_demo/result/new_dataset/test/'.

* python detect.py [--weights ./yolov5_detection/runs/train/exp/weight/best.pth] [--source ./dataset/test/] [--project ./3VV_demo/result/new_dataset/test/]
* python main2segmentation.py [--arch deeplabv3_gai] [--dataset MyDataset] [--epoch 35] [--batch_size 16] <br />

The first line of code is for detecting ROI areas and performing cropping. The second line of code is for segmenting the three blood vessels.

****
## **For training and testing：** <br /> 
If you wish to retrain the model on your own dataset, please follow the steps below to configure the parameters.  Before training, make sure to prepare your dataset in the following format: <br /> 

![image](https://github.com/ylfas/3VV_demo/assets/110209878/40c8a67d-aaff-495d-bcd3-c9a3f339c994) <br />

### **First stage：** <br />
* 1.Set the path for the detection dataset in the '.../yolov5_detection/data/ab.yaml' file. <br /> 

* 2.Configure the paths in the '.../yolov5_detection/train.py' file and set the number of epochs according to the task. <br />

* 3.Run 'python train.py' for detection training. <br />

* 4.After training completion, in '.../yolov5_detection/detect.py', input the path of the trained weight file in '--weights'. <br />

* 5.In detect.py, adjust the thresholds of '--conf-thres' and '--iou-thres' to achieve a more comprehensive detection effect. <br />

* 6.In detect.py, set the '--project' to save the predicted images. <br />

* 7.Run 'python detect.py' to complete ROI prediction on the test set. <br />

### **Second stage：**
* 1.Configure paths in '.../3VV_demo/utils/detec_ROI_trim.py': 'predict_txt_path' represents the path to the yolov5 detection result txt file, and 'full_size_img_path' denotes the path to the predicted results, i.e., the save path set in '--project' for predicted images. <br />

* 2.In detec_ROI_trim.py, set 'full_size_img_path_all' as the path to the initial full-size dataset, where the new test set for ROI extraction is obtained by cropping on the test set of this dataset. Finally, specify the path for storing the txt file recording the cropping coordinates ('.../3VV_demo/result/new_dataset/test/txt/'). <br />

* 3.In detec_ROI_trim.py, specify the paths for saving the cropped images and labels ('.../3VV_demo/result/new_dataset/test/img(mask)/'). <br />

* 4.Next, configure paths in '.../3VV_demo/utils/label_ROI_trim.py', including the paths to the training and validation sets of the initial full-size dataset, to generate new training and test sets for the segmentation network in the second stage. <br />

* 5.Finally, based on the paths of the newly generated datasets mentioned above, configure the paths for the new dataset in the MyDataset_trim class in '.../3VV_demo/dataset.py'. <br />

* 6.In '.../3VV_demo/main2segmentation.py', set the save path for predicted images ('.../3VV_demo/result/2th_stage_seg/deeplabv3/') and the txt path required for restoring image sizes ('.../3VV_demo/result/new_dataset/test/txt/'). <br />

* 7.Run 'python main2segmentation.py' to predict segmentation results in the second stage. <br />

* 8.In '.../3VV_demo/trim_predict.py', configure paths based on the segmentation prediction results path ('.../3VV_demo/result/2th_stage_seg/deeplabv3/') and the original full-size dataset path ('.../data/test/mask/'). <br />

* 9.Run 'python trim_predict.py' to complete the final evaluation of the 3VV results. <br />

****

**Yolov5_detection is the ROI detection model training for yolov5: The code is from https://github.com/ultralytics/yolov5； <br />**
