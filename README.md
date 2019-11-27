# PyTorch-YOLOv3
YOLOv3 trained on SVHN dataset (Number object detection)   
<br />
This repo is fork from https://github.com/eriklindernoren/PyTorch-YOLOv3 for finishing HW3     
of the Selected Topics in Visual Ricognition using DeepLearning course.  
<br />
The original owner of the repo has finished the training, inference and detection part.  
And I add some functions that can make it more easily trained on the SVHN dataset.
<br />
#### WHAT I HAVE DONE:


* In function `construct_train_data` of `constructData.py`   
The function can read the trainning data from the .mat file of the SVHN dataset,  
then convert it to the format that can be trained on the YOLOv3.  
Each line in the format define a bounding box in `<calss number> <x_center> <y_center> <width> <height>`.  
The coordinates is scaled between `[0, 1]`,   
and the image `data/custom/images/1.png` has the label path `data/custom/labels/1.txt`.  

* In `parseJson.py`  
Detect the img and store the detection result in json format,  
which is a list of dictionaries and each dictionary represent an img detection result in the following structure:  
   ```
   {
     "bbox": [7, 40, 40, 60],   => y1 x1 y2 x2 of bounding box
     "label" [5],               => class
     "score": [0.9]             => confidence
   }
   ```

For more information about training, inference and detection, please reference to the oringinal repo.
