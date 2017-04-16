import threading
import time
import urllib
from ast import literal_eval

import numpy as np
import tornado
import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.options import define, options, parse_command_line

import cv2
from web_demo import handle_one


define("port", default=8888, help="run on the given port", type=int)

# from http://stackoverflow.com/a/21844162
def ip_cam_stream(mjpeg_stream):
    ''' yields frames from an mjpeg stream '''
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

# seems like GPU pytorch model can't be called from multiple threads at once,
# TODO: make this batch together requests sent within 100-200ms of each other
py_t_lock = threading.Lock()
def serial_handle_one(*args, **kwargs):
    ''' synchronised version of handle_one '''
    with py_t_lock:
        return handle_one(*args, **kwargs)

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        self.cam_url = tornado.escape.url_unescape(self.get_argument("cam_url"))
        self.resize_to = tornado.escape.url_unescape(self.get_argument("resize_to"))
        self.resize_to = literal_eval('(' + self.resize_to + ')')
        self.stream.set_nodelay(True)
        self.closed = False

        def task():
            print self.cam_url
            for frame in ip_cam_stream(self.cam_url):
                #frame = cv2.imread('/home/studio/Desktop/ppltest.jpg')
                t = time.time()
                if self.resize_to is not None:
                    frame = cv2.resize(frame, self.resize_to)
                bones = serial_handle_one(frame, dont_draw=True)
                if len(bones['found_ppl']) == 0:
                    continue
                if self.closed:
                    break
                print self.cam_url, 'took: ', time.time() - t

                self.write_message(
                    bones
                )

            print 'closed'
        
        self.t = threading.Thread(target=task)
        self.t.setDaemon(True)
        self.t.start()
        
    def check_origin(self, origin):
        return True
        
        
    def on_close(self):
        self.closed = True

app = tornado.web.Application([
    (r'/bones', WebSocketHandler),
])

print app
parse_command_line()
print 'warming up'
_ = serial_handle_one(np.ones((1, 320,320,3)), dont_draw=True)
app.listen(options.port)
print 'ready'
tornado.ioloop.IOLoop.instance().start()
