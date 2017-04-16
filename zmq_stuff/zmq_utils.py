import numpy as np
import ujson
import zmq


def send_ndarray(sock, arr, extra='', flags=0):
    meta = dict(
        extra=extra,
        dtype=str(arr.dtype),
        shape=arr.shape, )
    sock.send_json(meta, flags | zmq.SNDMORE)
    return sock.send(arr, flags)


def send_ndarray_multipart(sock, arr, topic, extra='', flags=0):
    meta = ujson.dumps({
        'dtype': str(arr.dtype),
        'shape': arr.shape,
        'extra': extra
    })
    return sock.send_multipart([topic, meta, arr])


def recv_ndarray_multipart(sock, flags=0):
    [topic, meta_str, arr_bytes] = sock.recv_multipart()
    meta = ujson.loads(meta_str)
    arr = np.frombuffer(arr_bytes, dtype=meta['dtype'])
    return (topic, meta, arr.reshape(meta['shape']))


def recv_ndarray_raw(sock):
    msg = sock.recv()
    strt = msg.find('}')
    json_str, rest = msg[:strt + 1], msg[strt + 1:]
    print json_str
    meta = ujson.loads(json_str)
    arr = np.frombuffer(rest, dtype=meta['dtype'])
    return arr.reshape(meta['shape']), meta['extra']


def recv_ndarray(sock, flags=0):
    meta = sock.recv_json(flags=flags)
    print meta
    msg = sock.recv(flags=flags)
    print 'recv!'
    arr = np.frombuffer(msg, dtype=meta['dtype']).copy()
    print 'arrrr'
    return arr.reshape(meta['shape']), meta['extra']
