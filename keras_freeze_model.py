import sys
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_io



K.set_learning_phase(0)
K.set_image_data_format('channels_last')

OUTPUT_FOLDER= sys.argv[2]
INPUT_MODEL = sys.argv[1]
OUTPUT_GRAPH = 'frozen_model.pb'
OUTPUT_NODE_PREFIX = 'output_node'
NUMBER_OF_OUTPUTS = 1


try:
    model = load_model(INPUT_MODEL)
except ValueError as err:
    print('Please check the input saved model file')
    raise err

output = [None]*NUMBER_OF_OUTPUTS
output_node_names = [None]*NUMBER_OF_OUTPUTS
for i in range(NUMBER_OF_OUTPUTS):
    output_node_names[i] = OUTPUT_NODE_PREFIX+str(i)
    output[i] = tf.identity(model.outputs[i], name=output_node_names[i])
print('Output Tensor names: ', output_node_names)


sess = K.get_session()
try:
    frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)    
    graph_io.write_graph(frozen_graph, OUTPUT_FOLDER, OUTPUT_GRAPH, as_text=False)
    print(f'Frozen graph ready for inference/serving at {OUTPUT_FOLDER}/{OUTPUT_GRAPH}')
except:
    print('Error Occured')