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

#ifndef _DENSENET_H_
#define _DENSENET_H_


Tensor DenseBlock(Graph* graph, Tensor input, int numLayers, int growthRate)
{
  Tensor t, last = input;
  for (int i = 0; i < numLayers; i++) {
    //t = graph->batchnorm(last);
    //t = graph->relu(t);
    t = graph->conv2d(last, 4 * growthRate, 1, 1, 1, 1, 0, 0, false);
    //t = graph->batchnorm(t);
    //t = graph->relu(t);
    t = graph->conv2d(t, growthRate, 3, 3, 1, 1, 1, 1, false);
    Tensor inputs[2];
    inputs[0] = last; inputs[1] = t;
    last = graph->concat(2, inputs);
  }
  return last;
}

Tensor Transition(Graph* graph, Tensor input, int outputSize)
{
  Tensor t = graph->conv2d(input, outputSize, 1, 1, 1, 1, 0, 0, true);
  t = graph->pool2d_avg(t, 2, 2, 2, 2, 0, 0, true);
  return t;
}

Graph* DenseNet121(Model* model)
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
  Tensor t = graph->conv2d(input, 64, 7, 7, 2, 2, 3, 3, false);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 1, 1);
  int numFeatures = 64;
  t = DenseBlock(graph, t, 6, 32);
  numFeatures = (numFeatures + 32 * 6) / 2;
  t = Transition(graph, t, numFeatures);
  t = DenseBlock(graph, t, 12, 32);
  numFeatures = (numFeatures + 32 * 12) / 2;
  t = Transition(graph, t, numFeatures);
  t = DenseBlock(graph, t, 24, 32);
  numFeatures = (numFeatures + 32 * 24) / 2;
  t = Transition(graph, t, numFeatures);
  t = DenseBlock(graph, t, 16, 32);
  t = graph->pool2d_avg(t, 7, 7, 1, 1, 0, 0);
  return graph;
}

#endif
