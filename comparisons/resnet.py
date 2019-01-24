from tensorflow import keras
from enum import Enum
import squeezenet as sqn

Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Input = keras.layers.Input
GlobalAvgPooling2D = keras.layers.GlobalAveragePooling2D
Model = keras.models.Model

class ResNetBuilder(object):
  def __init__(self, input_format):
    self.input_format = input_format
    if (input_format == sqn.SqueezeNetInputFormat.CHANNELS_FIRST):
      self.input_format_string = "channels_first"
      self.concat_axis = 1
    else:
      self.input_format_string = "channels_last"
      self.concat_axis = 3
    self.module_constructors = [self.build_residual_module_0,
                                self.build_residual_module_1,
                                self.build_residual_module_2,
                                self.build_residual_module_3,
                                self.build_residual_module_4,
                                self.build_residual_module_5,
                                self.build_residual_module_6,
                                self.build_residual_module_7,
                                self.build_residual_module_8,
                                self.build_residual_module_9,
                                self.build_residual_module_10,
                                self.build_residual_module_11,
                                self.build_residual_module_12,
                                self.build_residual_module_13,
                                self.build_residual_module_14,
                                self.build_residual_module_15]

  def get_input_shape(self):
    return self.calc_dim_ordering(224, 224, 3)

  def calc_dim_ordering(self, h, w, c):
    if (self.input_format == sqn.SqueezeNetInputFormat.CHANNELS_FIRST):
      return (c, h, w)
    else:
      return (h, w, c)

  def get_num_channels(self, shape):
    if (self.input_format == sqn.SqueezeNetInputFormat.CHANNELS_FIRST):
      return shape[0]
    else:
      return shape[2]

  def get_height(self, shape):
    if (self.input_format == sqn.SqueezeNetInputFormat.CHANNELS_FIRST):
      return shape[1]
    else:
      return shape[0]

  def get_width(self, shape):
    if (self.input_format == sqn.SqueezeNetInputFormat.CHANNELS_FIRST):
      return shape[2]
    else:
      return shape[1]

  def get_residual_module_subgraph(self, residual_module_number):
    if residual_module_number < 0 or residual_module_number > 16:
      print("The valid residual modules are 1 through 15.")
      assert(0) 
    (inp, output) = self.module_constructors[residual_module_number](None)
    return Model(inp, output)

  def get_resnet_graph(self):
    inp_shape = self.get_input_shape()
    inp = Input(shape=inp_shape)
    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
    maxpool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', data_format=self.input_format_string)(conv1)
    _, res0 = self.build_residual_module_0(maxpool1)
    _, res1 = self.build_residual_module_1(res0)
    _, res2 = self.build_residual_module_2(res1)
    _, res3 = self.build_residual_module_3(res2)
    _, res4 = self.build_residual_module_4(res3)
    _, res5 = self.build_residual_module_5(res4)
    _, res6 = self.build_residual_module_6(res5)
    _, res7 = self.build_residual_module_7(res6)
    _, res8 = self.build_residual_module_8(res7)
    _, res9 = self.build_residual_module_9(res8)
    _, res10 = self.build_residual_module_10(res9)
    _, res11 = self.build_residual_module_11(res10)
    _, res12 = self.build_residual_module_12(res11)
    _, res13 = self.build_residual_module_13(res12)
    _, res14 = self.build_residual_module_14(res13)
    _, res15 = self.build_residual_module_15(res14)
    avgpool1 = GlobalAvgPooling2D(data_format=self.input_format_string)(res15)
    fc1 = keras.layers.Dense(1000, activation="softmax", use_bias=True)(avgpool1)
    return Model(inp, fc1)

  def build_residual_module(self, inp, inp_shape, downsample, module_number):
    if inp is None:
      if inp_shape is None or len(inp_dims) != 3:
        print("Input dimensions of (c,h,w)/(w,h,c) must be specified if no input tensor is given")
      inp = Input(shape=inp_dims)
    channels = self.get_num_channels(inp_shape)
    if (downsample):
      channels = channels * 2
      conv1 = Conv2D(channels, (3, 3), strides=(2, 2), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
    else:
      conv1 = Conv2D(channels, (3, 3), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
    conv2 = Conv2D(channels, (3, 3), padding='same', activation=None, use_bias=True, data_format=self.input_format_string)(conv1)
    if (downsample):
      bypass = Conv2D(channels, (1, 1), strides=(2, 2), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
    else:
      bypass = inp
    add = keras.layers.add([bypass, conv2], name=("out_" + str(module_number)))
    output = keras.layers.Activation("relu")(add)
    return (inp, output)

  def build_residual_module_0(self, inp):
    inp_shape = self.calc_dim_ordering(56, 56, 64)
    return self.build_residual_module(inp, inp_shape, False, 0)

  def build_residual_module_1(self, inp):
    inp_shape = self.calc_dim_ordering(56, 56, 64)
    return self.build_residual_module(inp, inp_shape, False, 1)

  def build_residual_module_2(self, inp):
    inp_shape = self.calc_dim_ordering(56, 56, 64)
    return self.build_residual_module(inp, inp_shape, False, 2)

  def build_residual_module_3(self, inp):
    inp_shape = self.calc_dim_ordering(56, 56, 64)
    return self.build_residual_module(inp, inp_shape, True, 3)

  def build_residual_module_4(self, inp):
    inp_shape = self.calc_dim_ordering(28, 28, 128)
    return self.build_residual_module(inp, inp_shape, False, 4)

  def build_residual_module_5(self, inp):
    inp_shape = self.calc_dim_ordering(28, 28, 128)
    return self.build_residual_module(inp, inp_shape, False, 5)

  def build_residual_module_6(self, inp):
    inp_shape = self.calc_dim_ordering(28, 28, 128)
    return self.build_residual_module(inp, inp_shape, False, 6)

  def build_residual_module_7(self, inp):
    inp_shape = self.calc_dim_ordering(28, 28, 128)
    return self.build_residual_module(inp, inp_shape, True, 7)

  def build_residual_module_8(self, inp):
    inp_shape = self.calc_dim_ordering(14, 14, 256)
    return self.build_residual_module(inp, inp_shape, False, 8)

  def build_residual_module_9(self, inp):
    inp_shape = self.calc_dim_ordering(14, 14, 256)
    return self.build_residual_module(inp, inp_shape, False, 9)

  def build_residual_module_10(self, inp):
    inp_shape = self.calc_dim_ordering(14, 14, 256)
    return self.build_residual_module(inp, inp_shape, False, 10)

  def build_residual_module_11(self, inp):
    inp_shape = self.calc_dim_ordering(14, 14, 256)
    return self.build_residual_module(inp, inp_shape, False, 11)

  def build_residual_module_12(self, inp):
    inp_shape = self.calc_dim_ordering(14, 14, 256)
    return self.build_residual_module(inp, inp_shape, False, 12)

  def build_residual_module_13(self, inp):
    inp_shape = self.calc_dim_ordering(14, 14, 256)
    return self.build_residual_module(inp, inp_shape, True, 13)

  def build_residual_module_14(self, inp):
    inp_shape = self.calc_dim_ordering(7, 7, 512)
    return self.build_residual_module(inp, inp_shape, False, 14)

  def build_residual_module_15(self, inp):
    inp_shape = self.calc_dim_ordering(7, 7, 512)
    return self.build_residual_module(inp, inp_shape, False, 15)
