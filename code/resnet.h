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

#ifndef _RESNET_H_
#define _RESNET_H_

Tensor BasicBlock2(Graph* graph, Tensor input, int outChannels, int stride)
{
  Tensor t1 = graph->conv2d(input, outChannels, 3, 3, stride, stride, 1, 1, true);
  t1 = graph->conv2d(t1, outChannels, 3, 3, 1, 1, 1, 1, false);
  Tensor t2;
  if (stride == 1) {
    t2 = input;
  } else {
    t2 = graph->conv2d(input, outChannels, 1, 1, stride, stride, 0, 0, false);
  }
  return graph->add(t1, t2);
}

Tensor BasicBlock3(Graph* graph, Tensor input, int outChannels, int stride)
{
  Tensor t1 = graph->conv2d(input, outChannels, 1, 1, stride, stride, 0, 0, true);
  t1 = graph->conv2d(t1, outChannels, 3, 3, 1, 1, 1, 1, true);
  t1 = graph->conv2d(t1, outChannels, 1, 1, 1, 1, 0, 0, false);
  Tensor t2;
  if (stride == 1) {
    t2 = input;
  } else {
    t2 = graph->conv2d(input, outChannels, 1, 1, stride, stride, 0, 0, true);
  }
  return graph->add(t1, t2);
}

Tensor BottleneckBlock(Graph* graph, Tensor input, int outChannels,
                       int bnChannels, int stride)
{
  Tensor t = graph->conv2d(input, bnChannels, 1, 1, 1, 1, 0, 0, true);
  //t = graph->batchnorm(t);
  t = graph->conv2d(t, bnChannels, 3, 3, stride, stride, 1, 1, true);
  //t = graph->batchnorm(t);
  t = graph->conv2d(t, outChannels, 1, 1, 1, 1, 0, 0, true);
  return t;
}

Graph* ResNet34(Model* model)
{
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 4;
  input.dim[0] = BATCH_SIZE;
  input.dim[1] = 3;
  input.dim[2] = 222;
  input.dim[3] = 222;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  input = graph->noop(input);
  Tensor t = graph->conv2d(input, 64, 7, 7, 2, 2, 3, 3, true);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 1, 1);
  for (int i = 0; i < 3; i++)
    t = BasicBlock2(graph, t, 64, 1);
  for (int i = 0; i < 4; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BasicBlock2(graph, t, 128, stride);
  }
  for (int i = 0; i < 6; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BasicBlock2(graph, t, 256, stride);
  }
  for (int i = 0; i < 3; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BasicBlock2(graph, t, 512, stride);
  }
  t = graph->pool2d_avg(t, 7, 7, 1, 1, 0, 0);
  return graph;
}

Graph* ResNet50(Model* model)
{
  Graph *graph = new Graph(model);
  Tensor input;
  input.numDim = 4;
  input.dim[0] = BATCH_SIZE;
  input.dim[1] = 3;
  input.dim[2] = 222;
  input.dim[3] = 222;
  input.op.guid = 0;
  input.op.ptr = NULL;
  input.idx = 0;
  input = graph->noop(input);
  Tensor t = graph->conv2d(input, 64, 7, 7, 2, 2, 3, 3, true);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 1, 1);
  for (int i = 0; i < 3; i++)
    t = BasicBlock3(graph, t, 64, 1);
  for (int i = 0; i < 4; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BasicBlock3(graph, t, 128, stride);
  }
  for (int i = 0; i < 6; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BasicBlock3(graph, t, 256, stride);
  }
  for (int i = 0; i < 3; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BasicBlock3(graph, t, 512, stride);
  }
  t = graph->pool2d_avg(t, 7, 7, 1, 1, 0, 0);
  return graph;
}
#endif
