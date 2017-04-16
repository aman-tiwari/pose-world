# Pose World

This is the raw code for http://golancourses.net/excap17/a/04/13/a-event/

**This relies entirely on [tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation). Reproduced below is the README.md at the time of this project.**

My modifications to it include optimising the handle_one function in [web_demo.py](web_demo.py) and adding [Websocket](http_demo.py) and [zmq](zmq_stuff/zmq_cv.py) servers that expose an api to be able to remotely use this. 

In the [js/](js/) folder is an example client for the Websocket server, in the [zmq_stuff](zmq_stuff) folder is the (now possibly somewhat broken!) zmq client and server. To use the client, make a `places.json` in the [js/](js/) folder, following the template:
```
{
    "place_name" : {
        "url":"mjpeg stream url",
        "corner": [top left corner of drawn camera, top right corner of drawn camera],
        "resize_to": [width to resize frames to before processing, height to resize frames to before processing],
        "colour": [r, g, b colour of drawn skeleton]
    },

}
```
An example `places.json` is included. 

## pytorch_Realtime_Multi-Person_Pose_Estimation
This is a pytorch version of Realtime_Multi-Person_Pose_Estimation, origin code is here https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation 

**This relies entirely on [tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation). Reproduced below is the README.md at the time of this project.**

## Introduction
Code repo for reproducing 2017 CVPR Oral paper using pytorch.  

## Contents
1. [Testing](#testing)
2. [Training](#training)

## Require
1. [Pytorch](http://pytorch.org/)
2. [Caffe](http://caffe.berkeleyvision.org/) is required if you want convert caffe model to a pytorch model.

## Testing
- `cd model; sh get_model.sh` to download caffe model or download converted pytorch model(https://www.dropbox.com/s/ae071mfm2qoyc8v/pose_model.pth?dl=0).
- `cd caffe_to_pytorch; python convert.py` to convert a trained caffe model to pytorch model. The converted model have relative error less than 1e-6, and will be located in `./model` after converted.
- `pythont picture_demo.py` to run the picture demo.
- `pythont web_demo.py` to run the web demo.

## Training
TODO

### Network Architecture
![Teaser?](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/readme/pose.png)

## Citation
Please cite the paper in your publications if it helps your research:    
	  
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }
