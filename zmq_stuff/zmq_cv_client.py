import cv2
import zmq
import numpy as np
import os
from zmq_utils import *


if 'zmq_port' in os.environ:
    PORT = os.environ['zmq_port']


def center_crop(arr, width, height):
    return arr[
        arr.shape[0] / 2 - width / 2: arr.shape[0] / 2 + width / 2,
        arr.shape[1] / 2 - height / 2: arr.shape[1] / 2 + height / 2,
        :
    ]


if __name__ == '__main__':
    import time

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.set_hwm(3)
    sock.connect(PORT)

    #video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(
        'http://81.149.56.38:8083/mjpg/video.mjpg')

    while True:
        t = time.time()

        ret, frame = video_capture.read()
        #frame = cv2.resize(center_crop(frame, 720, 720), (320, 320))
        #frame = cv2.resize(center_crop(frame, 720, 720), (320, 320))

        #send_ndarray(sock, frame)
        #canvas, _ = recv_ndarray(sock)

        #canvas = cv2.resize(canvas, (640, 640))
        canvas = frame.copy()
        cv2.imshow('aaa', canvas)
        print 'tot: ', time.time() - t
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
