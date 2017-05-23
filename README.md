# Pose World & [Superboneworld](https://vimeo.com/217268832)

This is the source code for [http://golancourses.net/excap17/a/04/13/a-event/](http://golancourses.net/excap17/a/05/11/a-final/)
and [http://golancourses.net/excap17/a/05/11/a-final/](http://golancourses.net/excap17/a/05/11/a-final/) & [Superboneworld](https://vimeo.com/217268832217268832)

**This is heavily based off [tensorboy/pytorch Realtime Multi-Person Pose Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation). Reproduced after this section is the README.md of it at the time of this project.**

My modifications to it include cleaning up the code, making it somewhat more modular and optimising the `handle_one` function in [infer.py](infer.py) that performs a single forward pass and connects the 'joint heatmap' into skeletons. It also no longer resizes images to be a square before processing them, since it is fully convlutional it can work with images of all sizes. Generally the processing time is dominated by the post-processing of the joint-heatmap into skeletons.

I have also created various servers, clients and scripts:

* [Websocket](http_demo.py) with [Javascript client](js/main.js). Currently this consumes an mjpeg stream and outputs the skeletons present in the stream in realtime, but would be trivial to modify to consume other video streams. 
* [ZMQ](zmq_stuff/zmq_cv.ppy) with [ZMQ client](zmq_stuff/zmq_sub_client.py). Currently this consumes an mjpeg stream as with the Websocket server. There is also a [REQ-REP client](zmq_stuff/zmq_cv_client.py) present.
* The script [vid_demo.py](vid_demo.py) processes video files into a json that contains a `frames` array that contains the bone positions for the skeletons (disambiguated between skeletons) in the video per frame. There is a [Javascript visualiser](js/vid_main.js) that can load this information. This is [Superboneworld](https://vimeo.com/217268832217268832). 

The pose jsons used for Superboneworld are available at: [https://drive.google.com/open?id=0B-7CCAaMeqWDN3N0Y1U1UjZlQm8](https://drive.google.com/open?id=0B-7CCAaMeqWDN3N0Y1U1UjZlQm8)

Untar the file inside the [js/](js/) folder such that you have a folder named `jsons` containing the frame-by-frame json description (i.e `js/jsons/A_AP_Ferg_-_Shabba_Explicit_ft._A_AP_ROCKY-iXZxipry6kE.json`, `js/jsons/JAK VYPADÁ MŮJ PARKOUR TRÉNINK #3 _ TARY-xgrWm_g8hno.json` etc...).

**This project relies heavily on [tensorboy/pytorch Realtime Multi-Person Pose Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation). Reproduced below is the README.md at the time of this project.**

## pytorch Realtime Multi-Person Pose Estimation
This is a pytorch version of Realtime Multi-Person Pose Estimation, origin code is here [ZheC/Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

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
- `pythont infer.py` to run the web demo.

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
