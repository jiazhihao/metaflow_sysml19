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

#ifndef _INCEPTION_H_
#define _INCEPTION_H_
#include "ops.h"
#include "subst_examples.h"

Tensor inceptionA(Graph* graph, Tensor input, int inputC, int channels)
{
  assert(inputC == input.dim[1]);
  Tensor t1 = graph->conv2d(input, 64, 1, 1, 1, 1, 0, 0, true);
  Tensor t2 = graph->conv2d(input, 48, 1, 1, 1, 1, 0, 0, true);
  t2 = graph->conv2d(t2, 64, 5, 5, 1, 1, 2, 2, true);
  Tensor t3 = graph->conv2d(input, 64, 1, 1, 1, 1, 0, 0, true);
  t3 = graph->conv2d(t3, 96, 3, 3, 1, 1, 1, 1, true);
  t3 = graph->conv2d(t3, 96, 3, 3, 1, 1, 1, 1, true);
  Tensor t4 = graph->pool2d_avg(input, 3, 3, 1, 1, 1, 1, true);
  t4 = graph->conv2d(t4, channels, 1, 1, 1, 1, 0, 0, true);
  Tensor inputs[2];
  inputs[0] = t1; inputs[1] = t2;
  Tensor t12 = graph->concat(2, inputs);
  inputs[0] = t3; inputs[1] = t4;
  Tensor t34 = graph->concat(2, inputs);
  inputs[0] = t12; inputs[1] = t34;
  return graph->concat(2, inputs);
}

Tensor inceptionB(Graph* graph, Tensor input)
{
  Tensor t1 = graph->conv2d(input, 384, 3, 3, 2, 2, 0, 0, true);
  Tensor t2 = graph->conv2d(input, 64, 1, 1, 1, 1, 0, 0, true);
  t2 = graph->conv2d(t2, 96, 3, 3, 1, 1, 1, 1, true);
  t2 = graph->conv2d(t2, 96, 3, 3, 2, 2, 0, 0, true);
  Tensor t3 = graph->pool2d_avg(input, 3, 3, 2, 2, 0, 0, false);
  Tensor inputs[2];
  inputs[0] = t1; inputs[1] = t2;
  Tensor t12 = graph->concat(2, inputs);
  inputs[0] = t12; inputs[1] = t3;
  return graph->concat(2, inputs);
}

Tensor inceptionC(Graph* graph, Tensor input, int channels)
{
  Tensor t1 = graph->conv2d(input, 192, 1, 1, 1, 1, 0, 0, true);
  Tensor t2 = graph->conv2d(input, channels, 1, 1, 1, 1, 0, 0, true);
  t2 = graph->conv2d(t2, channels, 1, 7, 1, 1, 0, 3, true);
  t2 = graph->conv2d(t2, 192, 7, 1, 1, 1, 3, 0, true);
  Tensor t3 = graph->conv2d(input, channels, 1, 1, 1, 1, 0, 0, true);
  t3 = graph->conv2d(t3, channels, 7, 1, 1, 1, 3, 0, true);
  t3 = graph->conv2d(t3, channels, 1, 7, 1, 1, 0, 3, true);
  t3 = graph->conv2d(t3, channels, 7, 1, 1, 1, 3, 0, true);
  t3 = graph->conv2d(t3, 192, 1, 7, 1, 1, 0, 3, true);
  Tensor t4 = graph->pool2d_avg(input, 3, 3, 1, 1, 1, 1, true);
  t4 = graph->conv2d(t4, 192, 1, 1, 1, 1, 0, 0, true);
  Tensor inputs[2];
  inputs[0] = t1; inputs[1] = t2;
  Tensor t12 = graph->concat(2, inputs);
  inputs[0] = t3; inputs[1] = t4;
  Tensor t34 = graph->concat(2, inputs);
  inputs[0] = t12; inputs[1] = t34;
  return graph->concat(2, inputs);
}

Tensor inceptionD(Graph* graph, Tensor input)
{
  Tensor t1 = graph->conv2d(input, 192, 1, 1, 1, 1, 0, 0, true);
  t1 = graph->conv2d(t1, 320, 3, 3, 2, 2, 0, 0, true);
  Tensor t2 = graph->conv2d(input, 192, 1, 1, 1, 1, 0, 0, true);
  t2 = graph->conv2d(t2, 192, 1, 7, 1, 1, 0, 3, true);
  t2 = graph->conv2d(t2, 192, 7, 1, 1, 1, 3, 0, true);
  t2 = graph->conv2d(t2, 192, 3, 3, 2, 2, 0, 0, true);
  Tensor t3 = graph->pool2d_max(input, 3, 3, 2, 2, 0, 0, false);
  Tensor inputs[3];
  inputs[0] = t1; inputs[1] = t2; inputs[2] = t3;
  return graph->concat(3, inputs);
}

Tensor inceptionE(Graph* graph, Tensor input)
{
  Tensor t1 = graph->conv2d(input, 320, 1, 1, 1, 1, 0, 0, true);
  Tensor t2 = graph->conv2d(input, 384, 1, 1, 1, 1, 0, 0, true);
  Tensor t2a = graph->conv2d(t2, 384, 1, 3, 1, 1, 0, 1, true);
  Tensor t2b = graph->conv2d(t2, 384, 3, 1, 1, 1, 1, 0, true);
  Tensor inputs[4];
  inputs[0] = t2a; inputs[1] = t2b;
  t2 = graph->concat(2, inputs);
  Tensor t3 = graph->conv2d(input, 448, 1, 1, 1, 1, 0, 0, true);
  t3 = graph->conv2d(t3, 384, 3, 3, 1, 1, 1, 1, true);
  Tensor t3a = graph->conv2d(t3, 384, 1, 3, 1, 1, 0, 1, true);
  Tensor t3b = graph->conv2d(t3, 384, 3, 1, 1, 1, 1, 0, true);
  inputs[0] = t3a; inputs[1] = t3b;
  t3 = graph->concat(2, inputs);
  Tensor t4 = graph->pool2d_max(input, 3, 3, 1, 1, 1, 1, false);
  t4 = graph->conv2d(t4, 192, 1, 1, 1, 1, 0, 0, true);
  inputs[0] = t1; inputs[1] = t2; inputs[2] = t3; inputs[3] = t4;
  return graph->concat(4, inputs);
}

Graph* InceptionV3(Model* model)
{
  printf("Create InceptionV3 graph.\n");
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 4;
  input.dim[0] = BATCH_SIZE;
  input.dim[1] = 3;
  input.dim[2] = 299;
  input.dim[3] = 299;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  input = graph->noop(input);
  Tensor t = graph->conv2d(input, 32, 3, 3, 2, 2, 0, 0, true);
  t = graph->conv2d(t, 32, 3, 3, 1, 1, 0, 0, true);
  t = graph->conv2d(t, 64, 3, 3, 1, 1, 1, 1, true);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 0, 0);
  t = graph->conv2d(t, 80, 1, 1, 1, 1, 0, 0, true);
  t = graph->conv2d(t, 192, 3, 3, 1, 1, 0, 0, true);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 0, 0);
  t = inceptionA(graph, t, 192, 32);
  t = inceptionA(graph, t, 256, 64);
  t = inceptionA(graph, t, 288, 64);
  t = inceptionB(graph, t);
  t = inceptionC(graph, t, 128);
  t = inceptionC(graph, t, 160);
  t = inceptionC(graph, t, 160);
  t = inceptionC(graph, t, 192);
  t = inceptionD(graph, t);
  t = inceptionE(graph, t);
  t = inceptionE(graph, t);
  t = graph->pool2d_avg(t, 8, 8, 1, 1, 0, 0);
  return graph;
}

/*
Tensor createCBR(Graph* graph, Tensor input, int outputC,
                 int kernelH, int kernelW,
                 int strideH, int strideW,
                 int padH, int padW)
{
  Tensor t= graph->conv2d(input, outputC, kernelH, kernelW, strideH,
                          strideW, padH, padW, false);
  t = graph->batchnorm(t);
  t = graph->relu(t);
  return t;
}

Graph* inceptionE(Model* model, int channels)
{
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 4;
  input.dim[0] = BATCH_SIZE;
  input.dim[1] = channels;
  input.dim[2] = 8;
  input.dim[3] = 8;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  input = graph->noop(input);
  Tensor t1 = createCBR(graph, input, 320, 1, 1, 1, 1, 0, 0);
  Tensor t2 = createCBR(graph, input, 384, 1, 1, 1, 1, 0, 0);
  Tensor t2a = createCBR(graph, t2, 384, 3, 3, 1, 1, 1, 1);
  Tensor t2b = createCBR(graph, t2, 384, 3, 3, 1, 1, 1, 1);
  Tensor inputs[4];
  inputs[0] = t2a; inputs[1] = t2b;
  t2 = graph->concat(2, inputs);
  Tensor t3 = createCBR(graph, input, 448, 1, 1, 1, 1, 0, 0);
  t3 = createCBR(graph, t3, 384, 3, 3, 1, 1, 1, 1);
  Tensor t3a = createCBR(graph, t3, 384, 3, 3, 1, 1, 1, 1);
  Tensor t3b = createCBR(graph, t3, 384, 3, 3, 1, 1, 1, 1);
  inputs[0] = t3a; inputs[1] = t3b;
  t3 = graph->concat(2, inputs);
  Tensor t4 = graph->pool2d_max(input, 3, 3, 1, 1, 1, 1, false);
  t4 = createCBR(graph, t4, 192, 1, 1, 1, 1, 0, 0);
  inputs[0] = t1; inputs[1] = t2; inputs[2] = t3; inputs[3] = t4;
  Tensor t = graph->concat(4, inputs);
  return graph;
}
*/
#endif
