from tensorflow import keras
from enum import Enum

Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Input = keras.layers.Input
GlobalAvgPooling2D = keras.layers.GlobalAveragePooling2D
Model = keras.models.Model

class SqueezeNetBypass(Enum):
  NO_BYPASS = 0
  SIMPLE_BYPASS = 1
  COMPLEX_BYPASS = 2

class SqueezeNetInputFormat(Enum):
  CHANNELS_FIRST = 0
  CHANNELS_LAST = 1

class SqueezeNetBuilder(object):
  def __init__(self, bypass, input_format, optimized):
    self.bypass = bypass
    self.input_format = input_format
    if (input_format == SqueezeNetInputFormat.CHANNELS_FIRST):
      self.input_format_string = "channels_first"
      self.concat_axis = 1
    else:
      self.input_format_string = "channels_last"
      self.concat_axis = 3
    self.optimized = optimized
    self.module_constructors = [self.build_fire_module_2,
                                self.build_fire_module_3,
                                self.build_fire_module_4,
                                self.build_fire_module_5,
                                self.build_fire_module_6,
                                self.build_fire_module_7,
                                self.build_fire_module_8,
                                self.build_fire_module_9]

  def get_input_shape(self):
    return self.calc_dim_ordering(223, 223, 3)

  def calc_dim_ordering(self, h, w, c):
    if (self.input_format == SqueezeNetInputFormat.CHANNELS_FIRST):
      return (c, h, w)
    else:
      return (h, w, c)

  # Follows the naming convention in the squeezenet paper
  def get_fire_module_subgraph(self, fire_module_number):
    if fire_module_number < 2 or fire_module_number > 9:
      print("The valid fire modules are 2 through 9.")
      assert(0) 
    (inp, output) = self.module_constructors[fire_module_number - 2](None)
    return Model(inp, output)

  def get_squeezenet_graph(self):
    inp_shape = self.get_input_shape()
    inp = Input(shape=inp_shape)
    conv1 = Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
    maxpool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', data_format=self.input_format_string)(conv1)
    _, fire2 = self.build_fire_module_2(maxpool1)
    _, fire3 = self.build_fire_module_3(fire2)
    _, fire4 = self.build_fire_module_4(fire3)
    maxpool4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', data_format=self.input_format_string)(fire4)
    _, fire5 = self.build_fire_module_5(maxpool4)
    _, fire6 = self.build_fire_module_6(fire5)
    _, fire7 = self.build_fire_module_7(fire6)
    _, fire8 = self.build_fire_module_8(fire7)
    maxpool8 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', data_format=self.input_format_string)(fire8)
    _, fire9 = self.build_fire_module_9(maxpool8)
    conv10 = Conv2D(1000, (1, 1), strides=(1, 1), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(fire9)
    avgpool10 = GlobalAvgPooling2D(data_format=self.input_format_string, name="final_pool")(conv10)
    return Model(inp, avgpool10)

  def build_fire_module(self, inp, bypass, inp_dims, s11, e11, e33, module_number):
    if self.optimized:
     return self. build_fire_module_optimized(inp, bypass, inp_dims, s11, e11, e33, module_number)
    else:
     return self. build_fire_module_unoptimized(inp, bypass, inp_dims, s11, e11, e33, module_number)

  def build_fire_module_optimized(self, inp, bypass, inp_dims, s11, e11, e33, module_number):
    assert(bypass == SqueezeNetBypass.NO_BYPASS)
    if inp is None:
      if inp_dims is None or len(inp_dims) != 3:
        print("Input dimensions of (c,h,w)/(w,h,c) must be specified if no input tensor is given")
      inp = Input(shape=inp_dims)
    squeeze_layer = Conv2D(s11, (1, 1), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
    expand_layer_3x3 = Conv2D(e11 + e33, (3, 3), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(squeeze_layer)
    return (inp, expand_layer_3x3)

  def build_fire_module_unoptimized(self, inp, bypass, inp_dims, s11, e11, e33, module_number):
    if inp is None:
      if inp_dims is None or len(inp_dims) != 3:
        print("Input dimensions of (c,h,w)/(w,h,c) must be specified if no input tensor is given")
      inp = Input(shape=inp_dims)
    squeeze_layer = Conv2D(s11, (1, 1), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
    expand_layer_1x1 = Conv2D(e11, (1, 1), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(squeeze_layer)
    expand_layer_3x3 = Conv2D(e33, (3, 3), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(squeeze_layer)
    concat = keras.layers.concatenate([expand_layer_1x1, expand_layer_3x3], axis=self.concat_axis)
    if bypass == SqueezeNetBypass.NO_BYPASS:
      return (inp, concat)
    else:
      if bypass == SqueezeNetBypass.SIMPLE_BYPASS:
        bypass = inp
      else:
        bypass = Conv2D(e11 + e33, (1, 1), padding='same', activation='relu', use_bias=True, data_format=self.input_format_string)(inp)
      output = keras.layers.add([bypass, concat], name=("out_" + str(module_number)))
      return (inp, output)

  # If the bypass setting for the network is simple or complex, return simple.
  # Otherwise, return no bypass.
  def get_bypass_simple(self):
    if self.bypass == SqueezeNetBypass.NO_BYPASS:
      return SqueezeNetBypass.NO_BYPASS
    else:
      return SqueezeNetBypass.SIMPLE_BYPASS

  # If the bypass setting for the network is complex, return complex.
  # Otherwise, return no bypass.
  def get_bypass_complex(self):
    if self.bypass == SqueezeNetBypass.COMPLEX_BYPASS:
      return SqueezeNetBypass.COMPLEX_BYPASS
    else:
      return SqueezeNetBypass.NO_BYPASS

  # These modules follow the naming confention in the paper, so the two refers
  # to the layer in the whole network.  This is the first fire module.
  def build_fire_module_2(self, inp):
    inp_shape = self.calc_dim_ordering(55, 55, 96)
    return self.build_fire_module(inp, self.get_bypass_complex(), inp_shape, 16, 64, 64, 2)

  def build_fire_module_3(self, inp):
    inp_shape = self.calc_dim_ordering(55, 55, 128)
    return self.build_fire_module(inp, self.get_bypass_simple(), inp_shape, 16, 64, 64, 3)

  def build_fire_module_4(self, inp):
    inp_shape = self.calc_dim_ordering(55, 55, 128)
    return self.build_fire_module(inp, self.get_bypass_complex(), inp_shape, 32, 128, 128, 4)

  def build_fire_module_5(self, inp):
    inp_shape = self.calc_dim_ordering(27, 27, 256)
    return self.build_fire_module(inp, self.get_bypass_simple(), inp_shape, 32, 128, 128, 5)

  def build_fire_module_6(self, inp):
    inp_shape = self.calc_dim_ordering(27, 27, 256)
    return self.build_fire_module(inp, self.get_bypass_complex(), inp_shape, 48, 192, 192, 6)

  def build_fire_module_7(self, inp):
    inp_shape = self.calc_dim_ordering(27, 27, 384)
    return self.build_fire_module(inp, self.get_bypass_simple(), inp_shape, 48, 192, 192, 7)

  def build_fire_module_8(self, inp):
    inp_shape = self.calc_dim_ordering(27, 27, 384)
    return self.build_fire_module(inp, self.get_bypass_complex(), inp_shape, 64, 256, 256, 8)

  def build_fire_module_9(self, inp):
    inp_shape = self.calc_dim_ordering(13, 13, 512)
    return self.build_fire_module(inp, self.get_bypass_simple(), inp_shape, 64, 256, 256, 9)
