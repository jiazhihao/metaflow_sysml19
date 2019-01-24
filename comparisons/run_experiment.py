import argparse
from enum import Enum
import numpy as np
from tensorflow import keras
import time
import resnet as rn
import squeezenet as sqn

class Models(Enum):
  INCEPTION = "inception"
  SQUEEZENET = "squeezenet"
  RESNET = "resnet"
  DENSENET = "densenet"


parser = argparse.ArgumentParser()
parser.add_argument("--trt", help="Whether to run with TensorRT optimizations", action="store_true")
parser.add_argument("--tfxla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--model", help="Which model to run", choices=[m.value for m in Models])
parser.add_argument("--subgraph", help="Which subgraph of the model to run (blank means whole model)", type=int, default=-1)
parser.add_argument("--snbypass", help="Type of bypass to use in in squeezenet", choices=["none", "simple", "complex"])
parser.add_argument("--channel_last", help="Run with the format h,w,c (defaults to c,h,w)", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--keras_plot", help="Name of file to output keras plot of the graph")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 10)", type=int, default=10)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 3)", type=int, default=3)
args = parser.parse_args()

if (args.channel_last):
  input_format = sqn.SqueezeNetInputFormat.CHANNELS_LAST
else:
  input_format = sqn.SqueezeNetInputFormat.CHANNELS_FIRST

if (args.model ==  Models.INCEPTION.value):
  print("model not implemented")
  assert(0)
elif (args.model ==  Models.SQUEEZENET.value):
  if (args.snbypass == "none"):
    snbuilder = sqn.SqueezeNetBuilder(sqn.SqueezeNetBypass.NO_BYPASS, input_format, True)
  elif (args.snbypass == "simple"):
    snbuilder = sqn.SqueezeNetBuilder(sqn.SqueezeNetBypass.SIMPLE_BYPASS, input_format, False)
  else:
    snbuilder = sqn.SqueezeNetBuilder(sqn.SqueezeNetBypass.COMPLEX_BYPASS, input_format, False)
  if (args.subgraph == -1):
    m = snbuilder.get_squeezenet_graph()
  else:
    m = snbuilder.get_fire_module_subgraph(args.subgraph)
elif (args.model ==  Models.RESNET.value):
  rnbuilder = rn.ResNetBuilder(input_format)
  if (args.subgraph == -1):
    m = rnbuilder.get_resnet_graph()
  else:
    m = snbuilder.get_residual_module_subgraph(args.subgraph)
elif (args.model ==  Models.DENSENET.value):
  print("model not implemented")
  assert(0)
else:
  print("You must specify a model.")
  assert(0)

if (args.trt):
# TODO: James please fill in how to do this"
  print("TensorRT not integrated yet")
  assert(0)

if (args.tfxla):
# TODO: Todd please fill in how to do this"
  print("TensorflowXLA is not integrated yet")
  assert(0)

# DO the actual running here, I think the printing should be done after running
input_shape = m.get_layer(index=0).get_input_shape_at(0)
input_data = np.random.rand(1, input_shape[1], input_shape[2], input_shape[3]).astype('f')
times = []
for i in range(args.discard_iter + args.iterations):
  t0 = time.time()
  print(m.predict(input_data).shape)
  t1 = time.time()
  print(str(t1 - t0) + " seconds")
  times.append(t1 - t0)

total = 0
for i in range(args.discard_iter, len(times)):
  total += times[i]

avg = total / (args.iterations)
print("Average time of the last " + str(args.iterations) + " iterations: " + str(avg) + " seconds")


if (args.keras_plot):
  keras.utils.plot_model(m, to_file=args.keras_plot, show_shapes=True)

if (args.print_tensorboard):
  tensorboard = keras.callbacks.TensorBoard(log_dir=args.print_tensorboard, write_graph=True, write_images=False)
  tensorboard.set_model(m)
# TODO: James and Todd should see whose way to print makes morse sense to use
