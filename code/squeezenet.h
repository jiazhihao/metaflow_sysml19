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

#ifndef _SQUEEZENET_H_
#define _SQUEEZENET_H_
#include "ops.h"
#include "subst_examples.h"

Tensor fire_complex(Graph* graph, Tensor input, int squeeze, int expand)
{
  Tensor t1 = graph->conv2d(input, squeeze, 1, 1, 1, 1, 0, 0, true);
  Tensor t1a = graph->conv2d(t1, expand, 3, 3, 1, 1, 1, 1, true);
  Tensor t1b = graph->conv2d(t1, expand, 1, 1, 1, 1, 0, 0, true);
  Tensor inputs[2];
  inputs[0] = t1a; inputs[1] = t1b;
  t1 = graph->concat(2, inputs);
  Tensor t2;
  if (input.dim[1] == t1.dim[1]) {
    t2 = input;
  }
  else {
    t2 = graph->conv2d(input, 2 * expand, 1, 1, 1, 1, 0, 0, true);
  }
  return graph->element(OpBase::OP_EW_ADD, t1, t2);
}

Graph* SqueezeNetComplex(Model* model)
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
  Tensor t = graph->conv2d(input, 96, 7, 7, 2, 2, 3, 3, true);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 0, 0);
  t = fire_complex(graph, t, 16, 64);
  t = fire_complex(graph, t, 16, 64);
  t = fire_complex(graph, t, 32, 128);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 0, 0);
  t = fire_complex(graph, t, 32, 128);
  t = fire_complex(graph, t, 48, 192);
  t = fire_complex(graph, t, 48, 192);
  t = fire_complex(graph, t, 64, 256);
  t = graph->pool2d_max(t, 3, 3, 2, 2, 0, 0);
  t = fire_complex(graph, t, 64, 256);
  t = graph->conv2d(t, 1000, 1, 1, 1, 1, 0, 0, true);
  t = graph->pool2d_avg(t, 13, 13, 1, 1, 0, 0);
  return graph;
}

#endif
