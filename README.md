# Documentation
This is a vehicle cut in detection model made to warn the driver in case of sudden cut in by two/three/four wheeler to avoid any harm to both parties.Uses YOLO and numpy 


# Dependencies
To run this model you'll need to install the following dependencies or modules through the 
'pip install "package name" command(install pip if not installed).

* opencv-python, numpy, playsound
* The yolov3.cfg  and coco.names and alarm.wav are provided in the repo by default.
* yolov3.weights file can be found to download here{https://www.kaggle.comdatasets/shivam316/yolov3-weights}
* Place these in your root directory to avoid  any errors

# How to use this? 

1. Clone repo in your working directory
2. Install and download the dependencies
3. Configure the code to load your own image/video file in OpenCV

* cap = cv2.VideoCapture('your file path(recommended that the file be in root directory)')

   
4. Execute the code
5. You can wait for the video capture to end, or alternatively press 'q'   on your keyboard to stop the execution 

