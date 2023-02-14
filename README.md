# AI Series on Aquatic Creatures Part1: Counting
Coral bleaching is becoming more and more serious, 
resulting in the gradual reduction of biodiversity near 
coral reefs. Our team hopes to record organisms and count 
the number through long-term monitoring and statistics, 
so as to monitor and reflect the current changes in 
the vicinity of coral reefs.

## Description
This project using AI object detector to 
detect the aquatic creatures, then counting the detected creature.

## Models
The Models must be [downloaded](https://drive.google.com/drive/folders/13BjuVBc6bTYutdx1YoeEtNK8MGj6jYbF?usp=sharing) and placed in the following path.
````
─ .idea
─ faster_rcnn/
    ├ lobster.pb
    └ ...
─ protos
─ utils
─ videos
─ yolov4/
    ├ lobster.weights
    └ ...
─ yolov4API
─ ...
````

## Environment
<ul>
<li>python >= 3.6.9</li>
<li>CUDA >= 10.2</li>
<li>cuDNN >= 7.5.6</li>
</ul>

## Requirements
Install packages in terminal 
````
pip install opencv-contrb-python
pip install tensorflow-gpu == 1.14
pip install numpy >= 1.15
````
local:\
`pip install -r requirements.txt`

## Demo
### object detector
You can test your object detector with object_detection_video.py\
`python object_detection_video.py --detector yolov4`
### object counting
You can counting objects with object_tracking_counting_test.py\
`python object_tracking_counting_test.py --detector yolov4 --output rs`
## Acknowledgements
NSTC-110-2634-F-019 -002 -
