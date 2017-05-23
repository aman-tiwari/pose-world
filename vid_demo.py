from __future__ import print_function
import time
import os

import ujson
from cv2 import resize

import skvideo.io
from infer import handle_one
import fire

def get_resize(frame, y=None, x=None):
    print(frame.shape)
    frame_y = frame.shape[0]
    frame_x = frame.shape[1]
    print(frame_y, frame_x)
    if x is None:
        x = float(frame_x) *  float(y) / float(frame_y)
        y = int(y)
        x = int(x)
        return (y, x)
    if y is None:
        y = float(frame_y) *  float(x) / float(frame_x)
        x = int(x)
        y = int(y)
        return (y, x)
    print(y, x)
    return (y, x)

def process(x=None, y=None, output_prefix='', *paths):
    for path in paths:
        videogen = skvideo.io.vreader(path)
        frame = videogen.next()
        y, x = get_resize(frame, y=y, x=x)
        print('processing {} with original shape {}, resizing to: {}, {}'.format(path, frame.shape, y, x))
        output = os.path.join(output_prefix, os.path.splitext(os.path.split(path)[-1])[0] + '.json')
        if os.path.isfile(output):
            print('skipping {}, file exists'.format(output))
            continue
        
        print('saving to {}'.format(output))
          
        tot = []
        t = time.time()
        try:
            for i, frame in enumerate(videogen):
                tot.append(handle_one(resize(frame, (x, y)), dont_resize=True, dont_draw=True))
                n = None if tot[-1] is None else len(tot[-1])
                print('took: {:.3f} {}'.format(time.time() - t, n))
                t = time.time()

                if i % 200 == 0:
                    print(i, n, ' ')
        except RuntimeError:
            print('RuntimeError for {}'.format(path))
            print('continuing...')
            pass
        with open(output, 'w') as jsonf:
            ujson.dump({'frames': tot}, jsonf)
    print('\ndone!\n')
    
if __name__ == '__main__':
    fire.Fire()