# synthetic-target-line-of-bearing

Create function to provide angle of bearing for person in an image. The following code was created using both the Nvidia and Tensorflow docker. The code should work if only using CPU but will still need to install tensorflow

## Discussion
The model chosen to perform object recognition was SSD: Single Shot MultiBox Detector https://research.google.com/pubs/pub44872.html. This model was chosen since it has low computation and is felixible to low resolution images to provide a near real time object dection. The COCO dataset was chosen for training since it is best for object recogition for small objects within image. The currenlt method to identify object angle uses the input parameter of line of bearing of camera and field of view of camera. Simple math operations are performed to derive object angle from the center of the image.

## Contribution
This is the first iteration of this project. Please feel free to contribute to this project by providing issue notificaitons, suggestions of methodology improvements, or suggestions of any kind. Please fork this project first before you make updates to this code. 

## Dependencies

#### Python Packages

* tensorflow
* numpy 
* os
* six.moves.urllib
* sys
* tarfile
* zipfile
* pandas
* math
* collections
* io
* matplotlib
* pil

#### Docker

* Tensorflow docker container https://github.com/tensorflow/tensorflow.git
* Nvidia docker https://github.com/NVIDIA/nvidia-docker
        - This will be needed if you want to use the gpu for this script
* need to update python package protobuf to 3.4.0 - I used code "pip install protobuf --upgrade"  

