import time
import json
import os

from settings import libPath

def timeit(method):
    "A decorator function for timing function execution."
    import time
    def timed(*args, **kwargs):
        ts = time.time()
        ret = method(*args, **kwargs)
        te = time.time()
        print('Done: ' + method.__name__.upper() + ' in %s seconds' % str(te-ts))
        return ret
    return timed

def read_params(path=libPath+'\\src\\cfg\\params.json'):
    "function that reads the training parameters from a config file"

    if os.path.exists(path):
        f = open(path)
    elif os.path.exists(libPath + '\\src\\cfg\\' + path):
        f = open(libPath + '\\src\\cfg\\' + path)
    else:
        print('File: ' + path + ' not found, using default instead')
        f = open(libPath + '\\src\\cfg\\params.json')

    data = json.load(f) # read the json formatted file.

    f.close() # close the opened file.

    return data