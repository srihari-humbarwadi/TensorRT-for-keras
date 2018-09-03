import os
import sys
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io

_input = <INPUT_TENSOR_NAME>
_output = <OUTPUR_TENSOR_NAME>
outputs = [_output]


def get_frozen_graph():
  with tf.gfile.FastGFile(sys.argv[1], "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def


frozen_graph_def = get_frozen_graph()
trt_graph_def = trt.create_inference_graph(frozen_graph_def, 
									outputs,
									max_batch_size=16, 
									max_workspace_size_bytes=2<<10<<20, 
									precision_mode='FP32')
tf.reset_default_graph()
g = tf.Graph()
with tf.Session(graph=g) as sess:
	with g.as_default():
		tf.import_graph_def(
  		graph_def=trt_graph_def,
  		name='')
	graph_io.write_graph(g, '.', 'trt_'+sys.argv[1], as_text=False)
