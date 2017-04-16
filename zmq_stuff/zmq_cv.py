import os
import zmq
import numpy as np
import time
import ujson
from web_demo import *
from zmq_utils import *

PORT = None

if 'zmq_port' in os.environ:
    PORT = os.environ['zmq_port']


import urllib


def ip_cam_stream(mjpeg_stream):
    stream = urllib.urlopen(mjpeg_stream)
    bytes = ''
    while True:
        bytes += stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]
            yield cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),
                               cv2.IMREAD_COLOR)


if __name__ == '__main__':
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)

    print 'warming up...'
    _ = handle_one(np.ones((320, 320, 3)))

    sock.bind(PORT)
    print 'ready! bound to', PORT

#    for frame in ip_cam_stream('http://81.149.56.38:8083/mjpg/video.mjpg'):

    for frame in ip_cam_stream('http://71.196.34.213:80/mjpg/video.mjpg?COUNTER'):
        # if frame.shape[0] > 700:
        #    frame = cv2.resize(frame.shape[0] / 2, frame.shape[1] / 2)
        canvas = handle_one(frame)
        send_ndarray_multipart(sock, canvas, 'uk_barber')

if __name__ == '__main__':
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)

    print 'warming up...'
    _ = handle_one(np.ones((320, 320, 3)))

    sock.bind(PORT)
    print 'ready! bound to', PORT
    while True:
        frame, _ = recv_ndarray(sock)
        print 'recvd!!!'
        t = time.time()
        canvas = handle_one(np.ascontiguousarray(frame))
        print 'oooo !!!'
        print 'took: ', time.time() - t
        #canvas = np.transpose(canvas, (2, 0, 1))
        send_ndarray(sock, np.ascontiguousarray(canvas, dtype=np.uint8))

if __name__ == '__main__':
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)

    print 'warming up...'
    _ = handle_one(np.ones((320, 320, 3)))

    sock.bind(PORT)
    print 'ready! bound to', PORT
    while True:
        frame, _ = recv_ndarray_raw(sock)
        print 'recvd!!!'
        t = time.time()
        canvas = handle_one(np.ascontiguousarray(frame))
        print 'oooo !!!'
        print 'took: ', time.time() - t
        canvas = np.transpose(canvas, (2, 0, 1))
        send_ndarray(sock, np.ascontiguousarray(canvas, dtype=np.uint8))
    # send_ndarray(sock, np.ascontiguousarray(canvas, dtype=np.uint8))
