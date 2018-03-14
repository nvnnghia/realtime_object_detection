## Foked form: https://github.com/GustavZ/realtime_object_detection
## Added KCF tracker to improve detection speed (from 25FPS to 35 FPS)

## demo video
Detection ussing only SSD: https://youtu.be/JLKqoV4vETY

Combine SSD and KCF: https://youtu.be/6pn6qUNScaU

# realtime_object_detection
Realtime object detection based on [Tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) with an extreme Focus on Performance. <br />
Specialized for `ssd_mobilenet` models.
<br />
<br />
## Getting Started:  
- create a copy of `config.sample.yml` called `config.yml`
- Optional: Change Parameters in `config.yml` to load other models or to modify input params.
- For example: If you are not interested in visualization: set `visualize` to `False`. <br />
- run `image_detection.py` for single test image detection
- run `object_detection.py` for realtime object detection
- Enjoy!
- To run the detection using combination of SSD and KCF tracker
    - Currently, Only support for single thread ('split_model = True' have not supported the combination yet)
    - change parametter 'use_kcf' in config.yml
    - build kcf lib: ./makefile.sh
    - run object_detection_kcf.py
    - Enjoy!
<br />

## About the Project
The Idea was to create a realtime capable object detection pipeline on various machines. <br />
Plug and play, ready to use without deep previous knowledge.<br /> <br />
The following work has been done based on the original API:
- Capturing frames of a Camera-Input using OpenCV in seperate thread to increase performance
- Calculate, print and optionally visualize current-local and total FPS
- Allows Models to grow GPU memory allocation. *(ssd_mobilenet_v11_coco needs 350 MB)*
- Added Option for detection without visualization to increase performance
- Added optional automated model download from [model-zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) if necessary
- Added `config.yml` for quick&easy parameter parsing
- Exported new frozen Model based on `ssd_mobilenet_v1_coco` with altered `batch_non_max_suppression.score_threshold` to increase perfomance
- Added various scripts in `/stuff` to make use of tf's API
- Added `split_model` Option to split frozen graph into a GPU and CPU session (Many thanks to [wkelongws](https://github.com/wkelongws)). <br />
Works only for `ssd_mobilenet` models but results in significant performance increase. 
- Added mutlithreading for the split sessions (Many thanks to [naisy](https://github.com/naisy))
- **Results: Overall up to x10 Performance Increase** depending on the config and the running system
<br />


## My Setup:
- Ubuntu 16.04
- Python 2.7
- Tensorflow 1.4
- OpenCV 3.3.1
 <br />

## Current max Performance on `ssd_mobilenet` (with|without visualization):
- Dell XPS 15 with i7 @ 2.80GHZ x8 and GeForce GTX 1050 4GB:  **78fps | 105fps**
- Nvidia Jetson Tx2 with Tegra 8GB:                           **30fps | 33 fps**
 <br />

