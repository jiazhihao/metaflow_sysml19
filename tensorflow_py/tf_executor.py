import argparse
import tensorflow as tf
import numpy as np
import time

CONST_VALUE = .00000182
GLOBAL_DATA_FORMAT = "NCHW"

class PaddingType:
  SAME = 0
  VALID = 1

class ActiMode:
  AC_MODE_NONE = 0
  AC_MODE_SIGMOID = 1
  AC_MODE_RELU = 2
  AC_MODE_TANH = 3

class OpType:
  OP_NOOP = 0
  OP_ANY = 1
  OP_CONV2D = 2
  OP_LINEAR = 3
  OP_POOL2D_MAX = 4
  OP_POOL2D_AVG = 5
  OP_RELU = 6
  OP_SIGMOID = 7
  OP_BATCHNORM = 8
  OP_CONCAT = 9
  OP_SPLIT = 10
  # RNN operators
  OP_EW_ADD = 11
  OP_EW_MUL = 12
  OP_MATMUL = 13

def get_padding_string(padding_type):
  if (padding_type == PaddingType.SAME):
    return "SAME"
  elif (padding_type == PaddingType.VALID):
    return "VALID"
  else:
    print("Unknown padding string")
    assert(0)

def split_string_ints(string, delim):
  ints = []
  splits = string.split(delim)
  for s in splits:
    ints.append(int(s))
  return ints

def split_string_int_pairs(string, delim1, delim2):
  pairs = []
  splits = string.split(delim1)
  for s in splits:
    pair_split = s.split(delim2)
    pairs.append((int(pair_split[0]), int(pair_split[1])))
  return pairs

def make_conv2d_with_bias(input_tensor, filter_shape, strides, padding, bias_dim, add_relu, name):
  weights_name = name + "_weights"
  bias_name = name + "_bias"
  conv_name = name + "_conv2d"
  bias_add_name = name + "_bias_add"

  weights = tf.constant(CONST_VALUE, shape=filter_shape, name=weights_name)
  bias = tf.constant(CONST_VALUE, shape=[bias_dim], name=bias_name)
  conv2d = tf.nn.conv2d(input_tensor, weights, strides, get_padding_string(padding), data_format=GLOBAL_DATA_FORMAT, name=conv_name)
  bias_add = tf.nn.bias_add(conv2d, bias, data_format=GLOBAL_DATA_FORMAT, name=bias_add_name)

  if (add_relu):
    relu_name = name + "_relu"
    relu = tf.nn.relu(bias_add, name=relu_name)
    return relu
  else:
    return bias_add

def create_input(line, operator_map):
  dims = split_string_ints(line, ',')
  input_shape = []
  for i in xrange(0, len(dims)):
    if (dims[i] > 0):
      input_shape.append(dims[i])
  input_placeholder = tf.placeholder(tf.float32, shape=input_shape)
  operator_map[(0,0)] = input_placeholder
  return input_shape

def parse_operator(line1, line2, line3, line4, operator_map, graph_outputs):
  guid = int(line1) 
  op_type = int(line2)
  deps = split_string_int_pairs(line3, ',', ':')
  for dep in deps:
    if dep in graph_outputs:
      graph_outputs.remove(dep)
  if (op_type == OpType.OP_CONV2D):
    params = split_string_ints(line4, ',')
    filter_shape = [params[5], params[6], params[1], params[4]]
    strides = [1, 1, params[7], params[8]]
    if (params[9] > 0 or params[10] > 0):
      padding = PaddingType.SAME
    else:
      padding = PaddingType.VALID
    name = "conv2d_" + str(guid)
    conv = make_conv2d_with_bias(operator_map[deps[0]], filter_shape, strides, padding, params[4], params[11], name=name)
    operator_map[(guid,0)] = conv
    return [(guid,0)]
  elif (op_type == OpType.OP_POOL2D_MAX or op_type == OpType.OP_POOL2D_AVG):
    params = split_string_ints(line4, ',')
    ksize = [1, 1, params[5], params[6]]
    strides = [1, 1, params[7], params[8]]
    if (params[9] > 0 or params[10] > 0):
      padding = PaddingType.SAME
    else:
      padding = PaddingType.VALID
    created_op = None
    name = None
    if (op_type == OpType.OP_POOL2D_MAX):
      name = "maxpool_" + str(guid)
      max_pool = tf.nn.max_pool(operator_map[deps[0]], ksize, strides, get_padding_string(padding), data_format=GLOBAL_DATA_FORMAT, name=name)
      created_op = max_pool
    else:
      name = "avgpool_" + str(guid)
      avg_pool = tf.nn.avg_pool(operator_map[deps[0]], ksize, strides, get_padding_string(padding), data_format=GLOBAL_DATA_FORMAT, name=name)
      created_op = avg_pool
    if params[10]: # If add relu
      relu_name = name + "_relu"
      relu = tf.nn.relu(created_op, name=relu_name)
      operator_map[(guid,0)] = relu
      return [(guid,0)]
    else:
      operator_map[(guid,0)] = created_op
      return [(guid,0)]
  elif (op_type == OpType.OP_SPLIT):
    params = split_string_ints(line4, ',')
    name = "split_" + str(guid)
    splits = tf.split(operator_map[deps[0]], params, 1, name=name)
    rets = []
    for i in xrange(0, len(splits)):
      operator_map[(guid, i)] = splits[i]
      rets.append((guid, i))
    return rets
  elif (op_type == OpType.OP_CONCAT):
    name = "concat_" + str(guid)
    dep_tensors = []
    for i in xrange(0, len(deps)):
      dep_tensors.append(operator_map[deps[i]])
    concat = tf.concat(dep_tensors, 1, name=name)
    operator_map[(guid,0)] = concat
    return [(guid,0)]
  elif (op_type == OpType.OP_EW_ADD):
    name = "ew_add_" + str(guid)
    ew_add = tf.add(operator_map[deps[0]], operator_map[deps[1]], name=name)
    operator_map[(guid,0)] = ew_add
    return [(guid,0)]
  elif (op_type == OpType.OP_EW_MUL):
    name = "ew_mul_" + str(guid)
    ew_mul = tf.multiply(operator_map[deps[0]], operator_map[deps[1]], name=name)
    operator_map[(guid,0)] = ew_mul
    return [(guid,0)]
  elif (op_type == OpType.OP_RELU):
    name = "relu_" + str(guid)
    relu = tf.nn.relu(operator_map[deps[0]], name=name)
    operator_map[(guid,0)] = relu
    return [(guid,0)]
  elif (op_type == OpType.OP_SIGMOID):
    name = "sigmoid_" + str(guid)
    sigmoid = tf.nn.sigmoid(operator_map[deps[0]], name=name)
    operator_map[(guid,0)] = sigmoid
    return [(guid,0)]
  elif (op_type == OpType.OP_BATCHNORM or op_type == OpType.OP_NOOP):
    operator_map[(guid,0)] = operator_map[deps[0]]
    return [(guid,0)]
  elif (op_type == OpType.OP_MATMUL):
    params = split_string_ints(line4, ',')
    assert(len(params) == 5)
    name = "matmul_" + str(guid)
    reshape_name = name + "_reshape"
    weights_name = name + "_weights"
    bias_name = name + "_bias"
    matmul_name = name + "_matmul"
    bias_add_name = name + "_bias_add"
    weights_shape = [params[2], params[3]]
    bias_dim = params[3]
    weights = tf.constant(CONST_VALUE, shape=weights_shape, name=weights_name)
    bias = tf.constant(CONST_VALUE, shape=[bias_dim], name=bias_name)
    reshape = tf.reshape(operator_map[deps[0]], [params[0]*params[1], params[2]], name=reshape_name)
    matmul = tf.matmul(reshape, weights, name=matmul_name)
    bias_add = tf.nn.bias_add(matmul, bias, name=bias_add_name)
    actimode = params[4]
    if (actimode == ActiMode.AC_MODE_NONE):
      operator_map[(guid,0)] = bias_add
      return [(guid,0)]
    elif (actimode == ActiMode.AC_MODE_SIGMOID):
      name += "_sigmoid"
      sigmoid = tf.nn.sigmoid(bias_add, name=name)
      operator_map[(guid,0)] = sigmoid
      return [(guid,0)]
    elif (actimode == ActiMode.AC_MODE_RELU):
      name += "_relu"
      relu = tf.nn.relu(bias_add, name=name)
      operator_map[(guid,0)] = relu
      return [(guid,0)]
    elif (actimode == ActiMode.AC_MODE_TANH):
      name += "_tanh"
      tanh = tf.nn.tanh(bias_add, name=name)
      operator_map[(guid,0)] = tanh
      return [(guid,0)]
    else:
      print("unknown actimode!!!!")
      assert(0)
  else:
    print("Found unknown opcode")
    assert(0)


parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--model_file", help="The file from which to load the model")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=50)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=10)
args = parser.parse_args()

input_shape = []
graph_outputs = set()

with open(args.model_file, 'r') as graph_file:
  # The graph nodes are repesented by 4 lines
  operator_map = {}
  need_input = True
  line1 = graph_file.readline()

  while line1:
    line2 = graph_file.readline()
    line3 = graph_file.readline()
    line4 = graph_file.readline()
    # Cut off the newlines
    line1 = line1[0:-1]
    line2 = line2[0:-1]
    line3 = line3[0:-1]
    line4 = line4[0:-1]

    if (need_input):
      need_input = False
      input_shape = create_input(line4, operator_map)
      graph_outputs.add((0,0))

    recent_outputs = parse_operator(line1, line2, line3, line4, operator_map, graph_outputs)
    for output in recent_outputs:
      graph_outputs.add(output)

    # Using this as the test of if the file is empty
    line1 = graph_file.readline()

if (len(graph_outputs) == 0):
  print("Could not read the graph!!!")
  assert(0)

output_nodes = []
for graph_output in graph_outputs:
  output_nodes.append(operator_map[graph_output])

config = tf.ConfigProto()
if (args.xla):
  config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

input_data = np.random.random_sample(input_shape)
with tf.Session(config=config) as sess:
  if (args.print_tensorboard):
    writer = tf.summary.FileWriter(args.print_tensorboard, sess.graph)
  times = []
  for i in range(args.discard_iter + args.iterations):
    t0 = time.time()
    sess.run(output_nodes, {operator_map[(0,0)]: input_data})
    t1 = time.time()
    print(str(t1 - t0) + " seconds")
    times.append(t1 - t0)

  total = 0
  for i in range(args.discard_iter, len(times)):
    total += times[i]
  avg = total / (args.iterations)
  print("Average time of the last " + str(args.iterations) + " iterations: " + str(avg) + " seconds")
