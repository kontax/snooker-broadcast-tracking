import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
videos = osp.join(this_dir, 'videos')
models = osp.join(this_dir, 'model')
caffe = osp.join(this_dir, 'py-faster-rcnn', 'caffe-fast-rcnn', 'python')
lib = osp.join(this_dir, 'py-faster-rcnn', 'lib')

add_path(videos)
add_path(models)
add_path(caffe)
add_path(lib)
