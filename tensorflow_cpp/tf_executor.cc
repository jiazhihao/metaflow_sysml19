// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <chrono>
using namespace std;


enum DataFormat {
  NCHW,
  NHWC,
};

enum PaddingType {
  SAME,
  VALID,
};

// This must be kept in sync with the type definitions in the graph code
// for now.
enum OpType {
  OP_NOOP, // 0
  OP_ANY, // 1
  OP_CONV2D, // 2
  OP_LINEAR, // 3
  OP_POOL2D_MAX, // 4
  OP_POOL2D_AVG, // 5
  OP_RELU, // 6
  OP_BATCHNORM, // 7
  OP_CONCAT, // 8
  OP_SPLIT, // 9
  // RNN operators
  OP_EW_ADD, // 10
  OP_EW_MUL, // 11
  OP_MATMUL, // 12
};


std::string get_data_format_string(DataFormat format)
{
  switch (format)
  {
    case NCHW:
    {
      return "NCHW";
    }
    case NHWC:
    {
      return "NHWC";
    }
    default:
    {
      abort();
      return "fuck compiler warnings";
    }
  }
}

std::string get_padding_type_string(PaddingType padding)
{
  switch (padding)
  {
    case SAME:
      return "SAME";
    case VALID:
      return "VALID";
    default:
      assert(0);
      return "fuck compiler warnings";
  }
}

tensorflow::TensorShape create_tensor_shape(std::vector<int> dims)
{
  using namespace tensorflow;
  TensorShape shape;
  for (int i = 0; i < dims.size(); i++)
  {
    shape.AddDim(dims[i]);
  }
  return shape;
}

// Want a "conv2d" maker that takes an input shape, input, weight shape, weights, and then creates the convolution with bias and activation

tensorflow::Output make_conv2d_with_bias(tensorflow::Input input,
                                         std::vector<int> filter_shape,
                                         std::vector<int> strides,
                                         DataFormat format,
                                         PaddingType padding,
                                         int bias_dim,
                                         bool add_relu,
                                         std::string name,
                                         tensorflow::Scope &scope)
{
  const float CONST_VALUE = .00000182;

  // First, let's make sure that things are being used correctly
  assert(input_shape == 4);
  assert(filter_shape.size() == 4);
  assert(strides.size() == 4);
  assert(strides[0] == 1);
  if (format == NCHW)
  {
    assert(strides[1] == 1);
  }
  else
  {
    assert(strides[3] == 1);
  }

  std::string weights_name(name);
  std::string bias_name(name);
  std::string conv_name(name);
  std::string bias_add_name(name);
  weights_name += "_weights";
  bias_name += "_bias";
  conv_name += "_conv2d";
  bias_add_name += "_bias_add";

  auto weights = tensorflow::ops::Const(scope.WithOpName(weights_name.c_str()), CONST_VALUE,
                                        create_tensor_shape(filter_shape));
  if (!scope.ok()) {
    LOG(FATAL) << scope.status().ToString();
    abort();
  }
  auto bias = tensorflow::ops::Const(scope.WithOpName(bias_name.c_str()), CONST_VALUE,
                                     {bias_dim});

  tensorflow::ops::Conv2D::Attrs attrs =
      tensorflow::ops::Conv2D::DataFormat("NCHW");

  auto conv2d = tensorflow::ops::Conv2D(scope.WithOpName(conv_name.c_str()), input,
                            weights, strides, get_padding_type_string(padding).c_str(), attrs);
  if (!scope.ok()) {
    LOG(FATAL) << scope.status().ToString();
    abort();
  }
  tensorflow::ops::BiasAdd::Attrs bias_attrs =
      tensorflow::ops::BiasAdd::DataFormat("NCHW");
  auto bias_add = tensorflow::ops::BiasAdd(scope.WithOpName(bias_add_name.c_str()), conv2d, bias, bias_attrs);
  if (!scope.ok()) {
    LOG(FATAL) << scope.status().ToString();
    abort();
  }

  if (add_relu)
  {
    std::string relu_name(name);
    relu_name += "_relu";
    auto relu_node =
        tensorflow::ops::Relu(scope.WithOpName(relu_name.c_str()), bias_add);
    if (!scope.ok()) {
      LOG(FATAL) << scope.status().ToString();
      abort();
    }
    return relu_node;
  }
  else
  {
    return bias_add;
  }
}

void split_string(const std::string &str,
                  char delimiter,
                  std::vector<std::string> &splits) {
  std::size_t from = 0;
  for (std::size_t i = 0; i < str.size(); ++i) {
    if (str[i] == delimiter) {
      splits.push_back(str.substr(from,i - from));
      from = i + 1;
    }
  }
  if (from < str.size())
  {
    splits.push_back(str.substr(from,str.size() - from));
  }
}

void split_string_ints(const std::string &str,
                       char delimiter,
                       std::vector<int> &splits) {
  std::vector<std::string> str_splits;
  split_string(str, delimiter, str_splits);
  for (unsigned i = 0; i < str_splits.size(); i++)
  {
    splits.push_back(atoi(str_splits[i].c_str()));
  }
}

void split_string_int_pairs(const std::string &str,
                            char delimiter1,
                            char delimiter2,
                            std::vector<std::pair<int,int> > &splits) {
  std::vector<std::string> str_splits;
  split_string(str, delimiter1, str_splits);
  for (unsigned i = 0; i < str_splits.size(); i++)
  {
    std::vector<std::string> second_splits;
    split_string(str_splits[i], delimiter2, second_splits);
    splits.push_back(
        std::make_pair(atoi(second_splits[0].c_str()),
                       atoi(second_splits[1].c_str())));
  }
}

void create_input(std::string &line,
                  tensorflow::Scope &scope,
                  DataFormat data_format,
                  std::map<std::pair<int, int>, tensorflow::Output> &operator_map)
{
  std::vector<int> shape;
  shape.push_back(1);
  split_string_ints(line, ',', shape);
  auto attrs =
      tensorflow::ops::Placeholder::Attrs().Shape(create_tensor_shape(shape));
  tensorflow::Output input = tensorflow::ops::Placeholder(
      scope.WithOpName("input"), tensorflow::DT_FLOAT, attrs);
  operator_map[std::make_pair(0,0)] = input;
}


tensorflow::OutputList parse_operator(std::string &line1,
    std::string &line2,
    std::string &line3,
    std::string &line4,
    tensorflow::Scope &scope,
    DataFormat data_format,
    std::map<std::pair<int, int>, tensorflow::Output> &operator_map)
{
  int guid = atoi(line1.c_str());
  int type = atoi(line2.c_str());

  std::vector<std::pair<int,int> > deps;
  split_string_int_pairs(line3, ',', ':', deps);

  switch(type)
  {
    case OP_CONV2D:
    { 
      std::vector<int> params;
      split_string_ints(line4, ',', params);
      std::vector<int> filter_shape;
      filter_shape.push_back(params[4]);
      filter_shape.push_back(params[5]);
      filter_shape.push_back(params[0]);
      filter_shape.push_back(params[3]);
      std::vector<int> strides;
      strides.push_back(1);
      strides.push_back(1);
      strides.push_back(params[6]);
      strides.push_back(params[7]);

      PaddingType padding;
      if (params[8] > 0 || params[9] > 0)
      {
        padding = SAME;
      }
      else
      {
        padding = VALID;
      }

      std::string name = "conv2d_";
      name += std::to_string(guid);

      tensorflow::Output conv = make_conv2d_with_bias(
          operator_map[deps[0]], filter_shape, strides,
          data_format, padding, params[3], params[10], "conv2d", scope);
      operator_map[std::make_pair(guid, 0)] = conv;
      return {conv};
    }
    case OP_POOL2D_MAX:
    case OP_POOL2D_AVG:
    {
      std::vector<int> params;
      split_string_ints(line4, ',', params);
      std::vector<int> ksize;
      ksize.push_back(1);
      ksize.push_back(1);
      ksize.push_back(params[4]);
      ksize.push_back(params[5]);
      std::vector<int> strides;
      strides.push_back(1);
      strides.push_back(1);
      strides.push_back(params[6]);
      strides.push_back(params[7]);

      PaddingType padding;
      if (params[8] > 0 || params[9] > 0)
      {
        padding = SAME;
      }
      else
      {
        padding = VALID;
      }

      if (type == OP_POOL2D_MAX)
      {
        std::string name = "maxpool_";
        name += std::to_string(guid);
        tensorflow::ops::MaxPool::Attrs attrs =
            tensorflow::ops::MaxPool::DataFormat("NCHW");
        tensorflow::Output max_pool = tensorflow::ops::MaxPool(
            scope.WithOpName(name.c_str()), operator_map[deps[0]],
            ksize, strides, get_padding_type_string(padding).c_str(), attrs);
        operator_map[std::make_pair(guid, 0)] = max_pool;
        if (!scope.ok()) {
          LOG(FATAL) << scope.status().ToString();
          abort();
        }
        return {max_pool};
      }
      else
      {
        std::string name = "avgpool_";
        name += std::to_string(guid);
        tensorflow::ops::AvgPool::Attrs attrs =
            tensorflow::ops::AvgPool::DataFormat("NCHW");
        tensorflow::Output avg_pool = tensorflow::ops::AvgPool(
            scope.WithOpName(name.c_str()), operator_map[deps[0]],
            ksize, strides, get_padding_type_string(padding).c_str(), attrs);
        operator_map[std::make_pair(guid, 0)] = avg_pool;
        if (!scope.ok()) {
          LOG(FATAL) << scope.status().ToString();
          abort();
        }
        return {avg_pool};
      }
    }
    case OP_SPLIT:
    {
      std::vector<int> params;
      split_string_ints(line4, ',', params);
      int num_splits = params.size();

      tensorflow::Tensor split_sizes(tensorflow::DT_INT32,
          tensorflow::TensorShape({4}));

      for (int i = 0; i < num_splits; i++)
      {
        split_sizes.vec<int>()(i) = params[i];
      }

      std::string name = "split_";
      name += std::to_string(guid);
      auto splits = tensorflow::ops::SplitV(
          scope.WithOpName(name.c_str()), operator_map[deps[0]],
          split_sizes, 1 /* split axis */, num_splits);

      for (int i = 0; i < num_splits; i++)
      {
        operator_map[std::make_pair(guid, i)] = splits[i];
      }
      return splits.output;
    }
    case OP_CONCAT:
    {
      std::string name = "concat_";
      name += std::to_string(guid);
      tensorflow::OutputList inputs_as_outputs;
      for (int i = 0; i < deps.size(); i++)
      {
        inputs_as_outputs.push_back(operator_map[deps[i]]);
      }
      tensorflow::Output concat = tensorflow::ops::Concat(
          scope.WithOpName(name.c_str()), inputs_as_outputs, 1 /* split axis */);

      operator_map[std::make_pair(guid, 0)] = concat;
      return {concat};
    }
    case OP_EW_ADD:
    {
      std::string name = "add_";
      name += std::to_string(guid);
      tensorflow::Output add = tensorflow::ops::Add(
          scope.WithOpName(name.c_str()),
          operator_map[deps[0]],
          operator_map[deps[1]]);

      operator_map[std::make_pair(guid, 0)] = add;
      return {add};
    }
    case OP_EW_MUL:
    {
      std::string name = "mul_";
      name += std::to_string(guid);
      tensorflow::Output mul = tensorflow::ops::Multiply(
          scope.WithOpName(name.c_str()),
          operator_map[deps[0]],
          operator_map[deps[1]]);

      operator_map[std::make_pair(guid, 0)] = mul;
      return {mul};
    }
    case OP_RELU:
    {
      std::string name = "relu_";
      name += std::to_string(guid);
      tensorflow::Output relu = tensorflow::ops::Relu(
          scope.WithOpName(name.c_str()),
          operator_map[deps[0]]);

      operator_map[std::make_pair(guid, 0)] = relu;
      return {relu};
    }
    case OP_BATCHNORM:
    case OP_NOOP:
    {
      operator_map[std::make_pair(guid, 0)] = operator_map[deps[0]];
      return {operator_map[deps[0]]};
    }
    case OP_MATMUL: // This doesn't seem to be implemented in run either
    default:
    {
      abort();
      return {operator_map[deps[0]]};
    }
  }
  
}


std::pair<tensorflow::Output, tensorflow::OutputList> buildTFGraphFromFile(
    tensorflow::Scope &scope,
    std::string file_name,
    DataFormat data_format)
{
  tensorflow::OutputList recent_outputs;
  std::map<std::pair<int, int>, tensorflow::Output> operator_map;

  std::string line1;
  std::string line2;
  std::string line3;
  std::string line4;
  ifstream myfile (file_name.c_str());
  if (myfile.is_open())
  {
    // create the placeholder for the input
    bool need_input = true;
    while ( getline(myfile,line1) )
    {
      // The format encodes nodes in 4 lines
      getline(myfile,line2);
      getline(myfile,line3);
      getline(myfile,line4);

      if (need_input)
      {
        need_input = false;
        create_input(line4, scope, data_format, operator_map);
      }

      recent_outputs = parse_operator(line1, line2, line3, line4, scope,
                     data_format, operator_map);
    }
    myfile.close();
  }
  return std::make_pair(operator_map[std::make_pair(0,0)], recent_outputs);
}


void print_timediff(const char* prefix, const struct timespec& start, const 
struct timespec& end)
{
      double milliseconds = (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3;
          printf("%s: %lf milliseconds\n", prefix, milliseconds);
}

int main(int argc, char **argv)
{
  bool use_xla = false;

  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"--xla")) {
      use_xla = true;
      continue;
    }
    fprintf(stderr, "Found unknown option!!\n");
    abort();
  }

  std::string FILE_NAME = "/home/users/twarszaw/tensorflow/tensorflow/cc/graph_transfer/tensorflow_cpp/test_squeeze";
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  std::pair<tensorflow::Output, tensorflow::OutputList> input_and_outputs =
      buildTFGraphFromFile(root, FILE_NAME.c_str(), NCHW);
  tensorflow::OutputList outputs = input_and_outputs.second;

  const float CONST_VALUE = .00000182;
  //TODO: read this starting shape from the input
  std::vector<int> starting_shape({1, 3, 222, 222});

  tensorflow::Input::Initializer input_tensor(CONST_VALUE,
                                  create_tensor_shape(starting_shape));

  std::vector<tensorflow::Tensor> output_tensors;

  tensorflow::SessionOptions sess_options;
  if (use_xla || true)
  {
    sess_options.config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(tensorflow::OptimizerOptions::ON_2);
  }
  fprintf(stderr, "opt level is: %d\n",
    sess_options.config.graph_options().optimizer_options().global_jit_level());
  tensorflow::ClientSession session(root, sess_options);
  //tensorflow::graph::SetDefaultDevice("/job:localhost/replica:0/task:0/device:XLA_GPU:0", session.impl()->

  tensorflow::Status s = session.Run({{input_and_outputs.first, input_tensor}}, outputs, &output_tensors);
  s = session.Run({{input_and_outputs.first, input_tensor}}, outputs, &output_tensors);
  s = session.Run({{input_and_outputs.first, input_tensor}}, outputs, &output_tensors);

  
  struct timespec start, end;
  double total_ms = 0.0;
  std::string cur_run_num;
  for (int i = 0; i < 10; i++)
  {
    clock_gettime(CLOCK_MONOTONIC, &start);
    s = session.Run({{input_and_outputs.first, input_tensor}}, outputs, &output_tensors);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double milliseconds = (end.tv_nsec - start.tv_nsec) / 1e6 + (end.tv_sec - start.tv_sec) * 1e3;
    total_ms += milliseconds;
    cur_run_num = "Run ";
    cur_run_num += std::to_string(i);
    print_timediff(cur_run_num.c_str(), start, end);
  }
  fprintf(stderr, "avg ms: %lf\n", total_ms/10);

  fprintf(stderr, "ran the graph\n");
  if (!s.ok())
  {
    fprintf(stderr, "error: %s\n", s.ToString().c_str());
    fprintf(stderr, "somethings fucked up\n");
    fprintf(stderr, "somethings fucked up\n");
    fprintf(stderr, "somethings fucked up\n");
    fprintf(stderr, "somethings fucked up\n");
    fprintf(stderr, "somethings fucked up\n");
  }

  return 0;

/*
  // This placeholder will not compile right now, but we will need it for later.
  //auto input = ops::placeholder(root.WithOpName("input"), DT_FLOAT, {-1, 56, 56, 96});

  // Matrix A = [3 2; -1 0]
  auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
  // Vector b = [3 5]
  auto b = Const(root, { {3.f, 5.f} });
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  */
}
