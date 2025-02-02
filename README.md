# RBE-549/Computer-Vision Final Project

1. This project uses Google’s MediaPipe library, if you do not have this library already installed please run the command:
```
pip install -q mediapipe==0.10
```
2. Depending on your environment configuration, there can be a conflict with MediaPipe and Tensorflow versions of protobuf. If you receive an error pertaining to a builder.py file, please follow the steps of the highest voted solution in this link: https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal
3. If using a virtual machine to run these scripts, the video feed will likely be laggy due to the intensive background computation. I recommend running this script on a local machine. Due to these limitations, I had to develop and run this program on my base Windows machine.
4. There is another script, utils.py, which the main script, reactions.py, relies on. The utils.py function contains some basic helper functions for running the main script.


In this project, the goal was to recreate and improvise existing and new reactions in Apple’s PhotoBooth app. There are eight existing reactions that are all triggered through hand gestures. Once a specific hand gesture is recognized, some type of visual effect will begin such as fireworks, rain, balloons, etc. The first part of this tasks involves implementing a Deep Learning technique for gesture recognition. The second part of this task involves creating the screen effects using different image processing techniques. These tasks can be further broken down into the following steps:
• Research and select a pre-trained model for gesture recognition and fine-tune model if necessary
• Run a webcam function that continuously reads frames from a webcam
• Pass each frame through the gesture recognition model and return the highest probability gesture
• Create visual effects (VFX) for each possible gesture that trigger when the respective gesture is recognized


Effects Implementation:
[![Watch the video](https://raw.githubusercontent.com/username/repository/branch/path/to/thumbnail.jpg)](https://github.com/mweissman2/RBE-549-Final-Project/blob/main/reactions_screen_recording.mp4)
