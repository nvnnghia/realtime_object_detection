#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:01:40 2017

@author: GustavZ
"""
import numpy as np
import os
import tensorflow as tf
import copy
import yaml
import cv2
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2
from deep_sort.tracker import Tracker
# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')
import threading
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from stuff.helper import FPS2, WebcamVideoStream, SessionWorker

from deep_sort import nn_matching
from timeit import default_timer as timer
from deep_sort.detection import Detection
import time

## LOAD CONFIG PARAMS ##
if (os.path.isfile('config_track.yml')):
    with open("config_track.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
else:
    print('no config file found')

video_input         = cfg['video_input']
visualize           = cfg['visualize']
vis_text            = cfg['vis_text']
max_frames          = cfg['max_frames']
width               = cfg['width']
height              = cfg['height']
fps_interval        = cfg['fps_interval']
allow_memory_growth = cfg['allow_memory_growth']
det_interval        = cfg['det_interval']
det_th              = cfg['det_th']
model_name          = cfg['model_name']
model_path          = cfg['model_path']
label_path          = cfg['label_path']
num_classes         = cfg['num_classes']
split_model         = cfg['split_model']
log_device          = cfg['log_device']
ssd_shape           = cfg['ssd_shape']
model 		    = cfg['model']
ratio 		    = cfg['ratio']
#preapare feature extraction
def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)

class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
        self.session = tf.Session(config=config)
        with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")
        self.input_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % input_name)
        self.output_var = tf.get_default_graph().get_tensor_by_name(
            "net/%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out
    def stopsess(self):
	self.session.close()
# Download Model form TF's Model Zoo
def download_model():
    model_file = model_name + '.tar.gz'
    download_base = 'http://download.tensorflow.org/models/object_detection/'
    if not os.path.isfile(model_path):
        print('Model not found. Downloading it now.')
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'toy_frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd() + '/models/')
        os.remove(os.getcwd() + '/' + model_file)
    else:
        print('Model found. Proceed.')

# helper function for split model
def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]

# Load a (frozen) Tensorflow model into memory.
def load_frozenmodel():
    print('Loading frozen model into memory')
    if not split_model:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return detection_graph, None, None

    else:
        # load a frozen Model and split it into GPU and CPU graphs
        # Hardcoded for ssd_mobilenet
        input_graph = tf.Graph()
        with tf.Session(graph=input_graph):
            if ssd_shape == 600:
                shape = 7326
            else:
                shape = 1917
            score = tf.placeholder(tf.float32, shape=(None, shape, num_classes), name="Postprocessor/convert_scores")
            expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name="Postprocessor/ExpandDims_1")
            for node in input_graph.as_graph_def().node:
                if node.name == "Postprocessor/convert_scores":
                    score_def = node
                if node.name == "Postprocessor/ExpandDims_1":
                    expand_def = node

        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']

            edges = {}
            name_to_node_map = {}
            node_seq = {}
            seq = 0
            for node in od_graph_def.node:
              n = _node_name(node.name)
              name_to_node_map[n] = node
              edges[n] = [_node_name(x) for x in node.input]
              node_seq[n] = seq
              seq += 1

            for d in dest_nodes:
              assert d in name_to_node_map, "%s is not in graph" % d

            nodes_to_keep = set()
            next_to_visit = dest_nodes[:]
            while next_to_visit:
              n = next_to_visit[0]
              del next_to_visit[0]
              if n in nodes_to_keep:
                continue
              nodes_to_keep.add(n)
              next_to_visit += edges[n]

            nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

            nodes_to_remove = set()
            for n in node_seq:
              if n in nodes_to_keep_list: continue
              nodes_to_remove.add(n)
            nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

            keep = graph_pb2.GraphDef()
            for n in nodes_to_keep_list:
              keep.node.extend([copy.deepcopy(name_to_node_map[n])])

            remove = graph_pb2.GraphDef()
            remove.node.extend([score_def])
            remove.node.extend([expand_def])
            for n in nodes_to_remove_list:
              remove.node.extend([copy.deepcopy(name_to_node_map[n])])

            with tf.device('/gpu:0'):
              tf.import_graph_def(keep, name='')
            with tf.device('/cpu:0'):
              tf.import_graph_def(remove, name='')

        return detection_graph, score, expand


def load_labelmap():
    print('Loading label map')
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


#video_stream = WebcamVideoStream(video_input,width,height).start()
cap = cv2.VideoCapture(video_input)
ret, first_frame = cap.read()
width = first_frame.shape[1]
height = first_frame.shape[0]

#gimage = video_stream.read()
gimage = cv2.resize(first_frame, (width/ratio, height/ratio)) 
gboxes1 =[]
isNotQuit = True
def detection(detection_graph, category_index, score, expand):
    print("Building Graph")
    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    cur_frames = 0
    global gboxes1
    global gimage
    global isNotQuit
    #image_encoder = ImageEncoder(model)
    #image_shape = image_encoder.image_shape
    #print ('load done! {}',image_shape)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=config) as sess:
            # Define Input and Ouput tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            if split_model:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                gpu_worker = SessionWorker("GPU",detection_graph,config)
                cpu_worker = SessionWorker("CPU",detection_graph,config)
                gpu_opts = [score_out, expand_out]
                cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]
                gpu_counter = 0
                cpu_counter = 0
            # Start Video Stream and FPS calculation
            fps = FPS2(fps_interval).start()
            
            cur_frames = 0
            print("Press 'q' to Exit")
            print('Starting Detection')
            #while video_stream.isActive():
	    while isNotQuit:
                # actual Detection
                if split_model:
                    # split model in seperate gpu and cpu session threads
                    if gpu_worker.is_sess_empty():
                        # read video frame, expand dimensions and convert to rgb
                        #image = video_stream.read()
			ret, image = cap.read()
			if not ret:
				isNotQuit = False
				break
			image = cv2.resize(image, (width/ratio, height/ratio)) 
			gimage = image.copy()
			boxes1 =[]
                        image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                        # put new queue
                        gpu_feeds = {image_tensor: image_expanded}
                        if visualize:
                            gpu_extras = image # for visualization frame
                        else:
                            gpu_extras = None
                        gpu_worker.put_sess_queue(gpu_opts,gpu_feeds,gpu_extras)

                    g = gpu_worker.get_result_queue()
                    if g is None:
                        # gpu thread has no output queue. ok skip, let's check cpu thread.
                        gpu_counter += 1
                    else:
                        # gpu thread has output queue.
                        gpu_counter = 0
                        score,expand,image = g["results"][0],g["results"][1],g["extras"]

                        if cpu_worker.is_sess_empty():
                            # When cpu thread has no next queue, put new queue.
                            # else, drop gpu queue.
                            cpu_feeds = {score_in: score, expand_in: expand}
                            cpu_extras = image
                            cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,cpu_extras)

                    c = cpu_worker.get_result_queue()
                    if c is None:
                        # cpu thread has no output queue. ok, nothing to do. continue
                        cpu_counter += 1
                        time.sleep(0.005)
                        continue # If CPU RESULT has not been set yet, no fps update
                    else:
                        cpu_counter = 0
                        boxes, scores, classes, num, image = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]
                else:
                    # default session
                    image = video_stream.read()
                    image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                    boxes, scores, classes, num = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_expanded})

                # Visualization of the results of a detection.
                if visualize:
		    bboxes = np.squeeze(boxes)
		    classes = np.squeeze(classes).astype(np.int32)
		    scores = np.squeeze(scores)
		    im_height, im_width, chanel = image.shape
                   # vis_util.visualize_boxes_and_labels_on_image_array(
                   #     image,
                    #    bboxes,
                    #    classes,
                    #    scores,
                    #    category_index,
                    #    use_normalized_coordinates=True,
                     #   line_thickness=8)
		    for i in range(bboxes.shape[0]):
			if category_index[classes[i]]['name'] =='person' and scores[i] > 0.5:
			    #print(bboxes[i])
			    #class_name = 
			    #if class_name == 'person':
		
			   	#print (class_name)
			    	#print(class_name)
			    	#print(100*scores[i])
			    	boxes1.append([(int)(bboxes[i][1]*im_width), (int)(bboxes[i][0]*im_height), (int)(bboxes[i][3]*im_width)- (int)(bboxes[i][1]*im_width), (int)(bboxes[i][2]*im_height)- (int)(bboxes[i][0]*im_height), scores[i] ])
			    #cv2.rectangle(image, ((int)(bboxes[i][1]*im_width),(int)(bboxes[i][0]*im_height)),((int)(bboxes[i][3]*im_width), (int)(bboxes[i][2]*im_height)), (0,255,0),3)
			    #cv2.putText(image ,class_name,((int)(bboxes[i][1]*im_width),(int)(bboxes[i][2]*im_height)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),3,cv2.LINE_AA)
                    #if vis_text:
                        #cv2.putText(image,"fps: {}".format(fps.fps_local()), (10,30),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                    #cv2.imshow('object_detection', image)
		    gboxes1 = boxes1
		    print ('====detect ', len(boxes1))
		    #print ('detect', boxes1)
                    # Exit Option
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    cur_frames += 1
                    # Exit after max frames if no visualization
                    for box, score, _class in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):
                        if cur_frames%det_interval==0 and score > det_th:
                            label = category_index[_class]['name']
                            print("label: {}\nscore: {}\nbox: {}".format(label, score, box))
                    if cur_frames >= max_frames:
                        break
                fps.update()

    # End everything
    if split_model:
    	gpu_worker.stop()
    	cpu_worker.stop()
    fps.stop()
    #video_stream.stop()
    cv2.destroyAllWindows()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))


def main():
    download_model()
    graph, score, expand = load_frozenmodel()
    category = load_labelmap()
    detection(graph, category, score, expand)

def extract_image_patch(image1, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image1.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image1 = image1[sy:ey, sx:ex]
    image1 = cv2.resize(image1, tuple(patch_shape[::-1]))
    return image1

def graph2():
    image_encoder = ImageEncoder(model)
    image_shape = image_encoder.image_shape
    print ('load done! {}',image_shape)
    global gboxes1
    global gimage
    #image = cv2.imread('11.png')
    #boxes1 = [[165, 111, 449, 366]]
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, None)
    tracker = Tracker(metric)
    results = []
    while(isNotQuit):
	image_patches = []
	detections = []
        io=0
	boxes1 = gboxes1
	graph_image = gimage.copy()
	if(len(boxes1)>0):
	   # print ('track', boxes1)
	   # print ('detect', len(boxes1))
	    for box1 in boxes1:
		    box = box1[0:4]
		    #print (box)
		   # cv2.imshow("image test", image)
		   # cv2.waitKey(1000)
		    patch = extract_image_patch(graph_image, box, image_shape[:2])
		    if patch is None:
			print("WARNING: Failed to extract image patch: %s." % str(box))
			patch = np.random.uniform(
			    0., 255., image_shape).astype(np.uint8)
		    image_patches.append(patch)
		    io+=1
	    print ('tracking', io)
	    image_patches = np.asarray(image_patches)
	    features =  image_encoder(image_patches, 32)
	    for i, feature in enumerate(features):
			detections.append(Detection(boxes1[i][0:4], boxes1[i][4], feature))
	    tracker.predict()
            tracker.update(detections)
	    for track in tracker.tracks:
		    if not track.is_confirmed() or track.time_since_update > 1:
		        continue
		    bbox = track.to_tlwh()
		    #results.append([
                    #frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
		    cv2.rectangle(graph_image, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,0,255), 2)
		    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(graph_image,str(track.track_id),(int(bbox[0]), int(bbox[1])),font,0.5,(0,255,255),1,cv2.LINE_AA)
		   # file1.write('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n' % (frame_count, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3])) 

	    boxes1 =[]
	cv2.imshow('video', graph_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    #image_encoder.stopsess()
    return 0

if __name__ == '__main__':
    thread1 = threading.Thread(target=main)
    thread2 = threading.Thread(target=graph2)
    thread1.start()
    thread2.start()
   # thread1.join()
   # thread2.join()
   #graph2()
    #main()
