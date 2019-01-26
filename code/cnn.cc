/* Copyright 2018 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops.h"
#include "subst_examples.h"
#include "inception.h"
#include "squeezenet.h"
#include "densenet.h"
#include "resnet.h"
#include "sru.h"
#include <cstring> 

int example(Model* model)
{
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 4;
  input.dim[0] = BATCH_SIZE;
  input.dim[1] = 384;
  input.dim[2] = 8;
  input.dim[3] = 8;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  Tensor t = graph->conv2d(input, 384, 3, 3, 1, 1, 1, 1, true);
  Tensor t1 = graph->conv2d(t, 384, 3, 3, 1, 1, 1, 1, true);
  Tensor t2 = graph->conv2d(t, 384, 3, 3, 1, 1, 1, 1, true);
}

Graph* optimize_graph(Graph *graph, Model *model, float alpha, int budget)
{
  std::vector<GraphXfer*> xfers;
  xfers.push_back(create_fuse_conv_batch_xfer(model));
  xfers.push_back(create_fuse_conv_relu_xfer(model));
  xfers.push_back(create_merge_conv_xfer(model));
  xfers.push_back(create_exclusive_concat_xfer(model));
  xfers.push_back(create_resnet_merge_xfer(model));

  std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare> candidates;
  std::set<size_t> hashmap;
  candidates.push(graph);
  hashmap.insert(graph->hash());
  Graph *bestGraph = graph;
  float bestCost = graph->total_cost();
  //printf("baselineCost = %.4lfms\n", bestCost);
  printf("Baseline Graph:\n    End-to-end runtime = %.4lfms\n", graph->run(model));
  graph->print_costs();

  int counter = 0;
  while (!candidates.empty()) {
    Graph *subGraph = candidates.top();
    candidates.pop();
    if (subGraph->total_cost() < bestCost) {
      delete bestGraph;
      bestCost = subGraph->total_cost();
      bestGraph = subGraph;
    }
    if (subGraph->total_cost() > alpha * bestCost) {
      delete subGraph;
      continue;
    }
    if (counter > budget) {
      // TODO: free all remaining candidates when budget exhausted 
      break;
    }
#ifdef VERBOSE
    if (counter % 100 == 0)
      printf("[%d] cost = %.4lf bestCost = %.4lf candidates.size() = %zu\n", counter, subGraph->total_cost(), bestCost, candidates.size());
#endif
    counter ++;
    for (int i = 0; i < xfers.size(); i++)
      xfers[i]->run(0, subGraph, candidates, hashmap, bestCost * alpha);
    if (bestGraph != subGraph) {
      delete subGraph;
    }
  }
  printf("Optimized Graph:\n    End-to-end runtime = %.4lfms\n", bestGraph->run(model));
  bestGraph->print_costs();

  return bestGraph;
}

enum DNNModel {
  None,
  SqueezeNet,
  Inception,
  Resnet34,
  Resnet50,
  DenseNet,
  RNNTC,
  NMT,
};

DNNModel name_to_model(std::string name)
{
  if (name == "inception") return Inception;
  if (name == "squeezenet") return SqueezeNet;
  if (name == "resnet34") return Resnet34;
  if (name == "resnet50") return Resnet50;
  if (name == "densenet") return DenseNet;
  if (name == "rnntc") return RNNTC;
  assert(false);
}

void parse_args(bool &optimize,
                bool &export_graph,
                float &alpha,
                int &budget,
                std::string &export_file_name,
                DNNModel &dnnModel,
                int argc,
                char **argv)
{
  std::string dnnName;
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i],"--noopt")) {
      optimize = false;
      continue;
    }
    if (!strcmp(argv[i], "--budget")) {
      budget = std::atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i],"--export")) {
      export_graph = true;
      export_file_name = argv[++i];
      continue;
    }
    if (!strcmp(argv[i],"--alpha")) {
      alpha = std::atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i],"--dnn")) {
      dnnName = std::string(argv[++i]);
      continue;
    }
/*
    if (!strcmp(argv[i],"--squeezenet")) {
      dnnModel = SqueezeNet;
      continue;
    }
    if (!strcmp(argv[i],"--inception")) {
      dnnModel = Inception;
      continue;
    }
    if (!strcmp(argv[i],"--resnet34")) {
      dnnModel = Resnet34;
      continue;
    }
    if (!strcmp(argv[i],"--resnet50")) {
      dnnModel = Resnet50;
      continue;
    }
    if (!strcmp(argv[i],"--densenet")) {
      dnnModel = DenseNet;
      continue;
    }
    if (!strcmp(argv[i],"--rnntc")) {
      dnnModel = RNNTC;
      continue;
    }
    if (!strcmp(argv[i],"--nmt")) {
      dnnModel = NMT;
      continue;
    }
*/
    fprintf(stderr, "Found unknown option!!\n");
    assert(0);
  }
  dnnModel = name_to_model(dnnName);
}

int main(int argc, char **argv)
{
  bool optimize = true;
  bool export_graph = false;
  int budget = 300; // 300 candidates
  float alpha = 1.01;
  DNNModel dnn = None;
  std::string export_file_name;
  parse_args(optimize, export_graph, alpha, budget, export_file_name, dnn, argc, argv);
  assert(dnn != None);
  printf("DnnModel(%d) alpha(%.4lf)\n", dnn, alpha);

  Model* model = new Model(false/*training*/);
  Graph* graph = NULL;
  switch (dnn) {
    case SqueezeNet:
      graph = SqueezeNetComplex(model);
      break;
    case Inception:
      graph = InceptionV3(model);
      break;
    case Resnet34:
      graph = ResNet34(model);
      break;
    case Resnet50:
      graph = ResNet50(model);
      break;
    case DenseNet:
      graph = DenseNet121(model);
      break;
    case RNNTC:
      graph = RNNTC_SRU(model);
      break;
    default:
      assert(false);
  }
  //Graph* graph = inceptionE(model, 2048);
#ifdef TRT
  void runGraphTRT(Graph *graph);
  //runGraphTRT(graph);
#endif
  if (optimize && dnn == RNNTC) {
    graph->print_costs();
    graph = RNNTC_OPT(model);
    printf("bestGraph: end-to-end runtime = %.2lf\n", graph->run(model));
    graph->print_costs();
  } else if (optimize) {
    graph = optimize_graph(graph, model, alpha, budget);
  }
  if (export_graph)
  {
    graph->export_to_file(export_file_name);
  }
#ifdef TRT
  runGraphTRT(graph);
#endif
  //inceptionE(model, 2048);
  //inceptionE(model, 1280);
  //inceptionD(model);
  //inceptionC(model, 192);
  //inceptionC(model, 160);
  //inceptionC(model, 128);
  //inceptionB(model);
  //inceptionA(model, 288, 64);
  //inceptionA(model, 256, 64);
  //inceptionA(model, 192, 32);
  return 0;
}
