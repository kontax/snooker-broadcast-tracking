from easydict import EasyDict


__C = EasyDict()
cfg = __C

__C.video_type = 'youtube'  # The lookup value for the video factory
__C.detection_frame = 5     # The frame number that detection is performed
__C.caffe_mode = 'GPU'      # Whether to use CPU or GPU for caffe
__C.gpu_device = 0          # If caffe_mode is GPU, then the GPU device number to use
__C.conf_threshold = 0.9    # The confidence threshold to keep predictions at
__C.nms_threshold = 0.3     # The threshold for non-maximum suppression
