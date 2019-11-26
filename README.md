# PyTorch-YOLOv3
YOLOv3 trained on SVHN dataset (Number object detection)   
<br />
This repo is fork from https://github.com/eriklindernoren/PyTorch-YOLOv3 for finishing HW3 in the NCTU's Selected Topics in Visual Ricognition using DeepLearning course.  
<br />
The original owner of the repo has finished the training, inference and detection part.  
And I add some functions that can make it more easily training on the SVHN dataset.
<br />
## WHAT I HAVE DONE:
* Read the trainning data from the .mat file of the SVHN dataset and convert it to the format that can trained by the YOLOv3,
  each line define a bounding box in `<calss number> <x_center> <y_center> <width> <height>`,  
  the coordinates is scaled between `[0, 1]`  
  the image `data/custom/images/1.png` has the label path `data/custom/labels/1.txt`
  
* Detect the img and store the detection result in json format  ,
  Which is a list of dictionaries and each dictionary represent an img detection result in the following structure:  
  <br />
  ```
  {
    "bbox": [7, 40, 40, 60],   => y1 x1 y2 x2 of bounding box
    "label" [5],               => class
    "score": [0.9]             => confidence
  }
  ```
