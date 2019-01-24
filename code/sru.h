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

#ifndef _SRU_H_
#define _SRU_H_
#include "ops.h"
#include "subst_examples.h"

#define EMBED_SIZE 1024

struct SRUTensors {
  Tensor c, h;
};

SRUTensors SRUNode(Graph* graph, Tensor x, Tensor c)
{
  Tensor x1 = graph->matmul(x, EMBED_SIZE);
  Tensor x2 = graph->matmul(x, EMBED_SIZE);
  Tensor f = graph->sigmoid(x2);
  Tensor x3 = graph->matmul(x, EMBED_SIZE);
  Tensor r = graph->sigmoid(x3);
  SRUTensors outputs;
  outputs.c = graph->add(graph->mul(f, c), graph->mul(f, x1));
  outputs.h = graph->add(graph->mul(r, outputs.c), graph->mul(r, x));
  //outputs.x = outputs.h;
  return outputs;
}

SRUTensors SRUOpt(Graph* graph, Tensor x, Tensor c)
{
  Tensor f = graph->matmul(x, EMBED_SIZE, OpBase::AC_MODE_SIGMOID);
  Tensor r = graph->matmul(x, EMBED_SIZE, OpBase::AC_MODE_SIGMOID);
  SRUTensors outputs;
  outputs.c = graph->mul(graph->add(c, r), f);
  outputs.h = graph->mul(r, graph->add(outputs.c, x));
  return outputs;
}

Graph* RNNTC_SRU(Model* model)
{
  const int LENGTH = 20;
  const int NUM_LAYERS = 1;
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 3;
  input.dim[0] = 1;
  input.dim[1] = BATCH_SIZE;
  input.dim[2] = EMBED_SIZE;
  input.dim[3] = 0;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  Tensor xs[LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    xs[i] = graph->noop(input);
  }
  Tensor c = graph->noop(input);
  SRUTensors sru[NUM_LAYERS][LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    if (i == 0)
      sru[0][i] = SRUNode(graph, xs[i], c);
    else
      sru[0][i] = SRUNode(graph, xs[i], sru[0][i-1].c);
  }
  return graph;
}

Graph* RNNTC_OPT(Model* model)
{
  const int LENGTH = 20;
  const int NUM_LAYERS = 1;
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 3;
  input.dim[0] = 1;
  input.dim[1] = BATCH_SIZE;
  input.dim[2] = EMBED_SIZE;
  input.dim[3] = 0;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  Tensor xs[LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    xs[i] = graph->noop(input);
  }
  Tensor c = graph->noop(input);
  SRUTensors sru[NUM_LAYERS][LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    if (i == 0)
      sru[0][i] = SRUOpt(graph, xs[i], c);
    else
      sru[0][i] = SRUOpt(graph, xs[i], sru[0][i-1].c);
  }
  return graph;
}

Graph* NMT_SRU(Model* model)
{
  const int LENGTH = 40;
  const int NUM_LAYERS = 2;
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 3;
  input.dim[0] = 1;
  input.dim[1] = BATCH_SIZE;
  input.dim[2] = EMBED_SIZE;
  input.dim[3] = 0;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  Tensor xs[LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    xs[i] = graph->noop(input);
  }
  Tensor c = graph->noop(input);
  SRUTensors sru[NUM_LAYERS][LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    for (int j = 0; j < NUM_LAYERS; j++) {
      Tensor x_in;
      if (i < LENGTH / 2)
        x_in = (j==0) ? xs[i] : sru[j-1][i].h;
      else
        x_in = (j==0) ? sru[NUM_LAYERS-1][i-1].h : sru[j-1][i].h;
      Tensor c_in = (i==0) ? c : sru[j][i-1].c;
      sru[j][i] = SRUNode(graph, x_in, c_in);
    }
  }
  return graph;
}

Graph* NMT_OPT(Model* model)
{
  const int LENGTH = 40;
  const int NUM_LAYERS = 2;
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 3;
  input.dim[0] = 1;
  input.dim[1] = BATCH_SIZE;
  input.dim[2] = EMBED_SIZE;
  input.dim[3] = 0;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  Tensor xs[LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    xs[i] = graph->noop(input);
  }
  Tensor c = graph->noop(input);
  SRUTensors sru[NUM_LAYERS][LENGTH];
  for (int i = 0; i < LENGTH; i++) {
    for (int j = 0; j < NUM_LAYERS; j++) {
      Tensor x_in;
      if (i < LENGTH / 2)
        x_in = (j==0) ? xs[i] : sru[j-1][i].h;
      else
        x_in = (j==0) ? sru[NUM_LAYERS-1][i-1].h : sru[j-1][i].h;
      Tensor c_in = (i==0) ? c : sru[j][i-1].c;
      sru[j][i] = SRUOpt(graph, x_in, c_in);
    }
  }
  return graph;
}

#endif
