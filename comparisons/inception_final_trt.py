import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib
import tensorflow.contrib.tensorrt as trt
import numpy as np
import time
from subprocess import call

K = keras.backend
Conv2D, MaxPooling2D, Input = keras.layers.Conv2D, keras.layers.MaxPooling2D, keras.layers.Input
Model = keras.models.Model

_GPU_MEM_FRACTION = 0.50
_WARMUP_NUM_LOOPS = 5

def get_gpu_config():
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=_GPU_MEM_FRACTION)
  return tf.ConfigProto(gpu_options=gpu_options)

def get_iterator(data):
  """Wrap numpy data in a dataset."""
  dataset = tf.data.Dataset.from_tensors(data).repeat()
  return dataset.make_one_shot_iterator()

def time_graph(graph_def, data, input_node, output_node, num_loops=10):
  tf.reset_default_graph()
  g = tf.Graph()

  with g.as_default():
    iterator = get_iterator(data)
    return_tensors = tf.import_graph_def(
        graph_def=graph_def,
        input_map={input_node: iterator.get_next()},
        return_elements=[output_node]
    )
    output = return_tensors[0].outputs[0]

  timings = []
  with tf.Session(graph=g, config=get_gpu_config()) as sess:
    call(["rm", "-rf", "trt"])
    tf.summary.FileWriter("trt").add_graph(sess.graph) # for tensorboard
    for _ in range(_WARMUP_NUM_LOOPS):
      sess.run([output])
    for _ in range(num_loops):
      tstart = time.time()
      val = sess.run([output])
      timings.append(time.time() - tstart)

  return timings, val[0]

def get_inception_model():
  inp = Input(shape=(2048, 8, 8))
  branch_1 = Conv2D(320, (1, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(inp)
  branch_2_3 = Conv2D(384, (1, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(inp)
  branch_2 = Conv2D(384, (1, 3), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_2_3)
  branch_3 = Conv2D(384, (3, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_2_3)
  branch_4_5 = Conv2D(448, (1, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(inp)
  branch_4_5 = Conv2D(384, (3, 3), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_4_5)
  branch_4 = Conv2D(384, (1, 3), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_4_5)
  branch_5 = Conv2D(384, (3, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_4_5)
  branch_6 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', data_format="channels_first")(inp)
  branch_6 = Conv2D(192, (1, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_6)
  output = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4, branch_5, branch_6], axis=1, name="out")
  return Model(inp, output)

def get_fire_module(complex_bypass):
  inp = Input(shape=(64 if complex_bypass else 128, 55, 55))
  branch_1_2 = Conv2D(16, (1, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(inp)
  branch_1 = Conv2D(64, (1, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_1_2)
  branch_2 = Conv2D(64, (3, 3), padding='same', activation='relu', use_bias=False, data_format="channels_first")(branch_1_2)
  concat = keras.layers.concatenate([branch_1, branch_2], axis=1)
  if complex_bypass:
    bypass = Conv2D(128, (1, 1), padding='same', activation='relu', use_bias=False, data_format="channels_first")(inp)
  else:
    bypass = inp
  output = keras.layers.add([bypass, concat], name="out")
  return Model(inp, output)


m = get_fire_module(True)
c, h, w = 64, 55, 55
out_name = "out/add"
# print([n.name for n in tf.get_default_graph().as_graph_def().node])

input_data = np.random.rand(1, c, h, w).astype('f')

# times for regular Keras
for i in range(10):
  t0 = time.time()
  print(m.predict(input_data).shape)
  print(str(time.time() - t0) + " seconds")

sess = K.get_session()
call(["rm", "-rf", "keras"])
tf.summary.FileWriter("keras").add_graph(sess.graph) # for tensorboard
saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
checkpoint_path = saver.save(sess, './saved_ckpt', global_step=0, latest_filename='checkpoint_state')
graph_io.write_graph(sess.graph, '.', 'tmp.pb')
freeze_graph.freeze_graph('./tmp.pb', '',
  False, checkpoint_path, out_name,
  "save/restore_all", "save/Const:0",
  './last_block.pb', False, "")

with tf.gfile.FastGFile('./last_block.pb', "rb") as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
orig_nodes = [n.name for n in graph_def.node]
print(orig_nodes)

trt_graph = trt.create_inference_graph(
                 input_graph_def = graph_def,
                 outputs = [out_name],
                 max_batch_size=1,
                 max_workspace_size_bytes=2<<30,
                 precision_mode="FP32")
trt_nodes = [n.name for n in trt_graph.node]
print(trt_nodes)

for n in orig_nodes:
  if n not in trt_nodes:
    print("orig not in trt: " + n)

for n in trt_nodes:
  if n not in orig_nodes:
    print("trt not in orig: " + n)

# times for tensorRT optimized
print(time_graph(trt_graph, input_data, "input_1", out_name)[0])
