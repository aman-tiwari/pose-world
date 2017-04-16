import cv2
import numpy as np
import zmq
import os
from zmq_utils import *

PORT = None

if 'zmq_port' in os.environ:
    PORT = os.environ['zmq_port']

if __name__ == '__main__':

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.set_hwm(3)
    sock.connect(PORT)
    sock.setsockopt(zmq.SUBSCRIBE, 'uk_barber')

    while True:
        topic, meta, processed = recv_ndarray_multipart(sock)
        cv2.imshow('p', processed.astype(np.uint8))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
