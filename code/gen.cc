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

#include <unordered_map>
#include "ops.h"
typedef int GENTYPE;
#define MAX_SIZE 100
#define MAX_NUM_TENSORS 6

struct TensorTemp {
  int numDim, dim[MAX_DIM];
  GENTYPE data[MAX_SIZE];
  std::string name;
  bool operator==(const TensorTemp& tt) const
  {
    if (tt.numDim != numDim) return false;
    int total = 1;
    for (int i = 0; i < numDim; i++) {
      if (dim[i] != tt.dim[i]) return false;
      total *= dim[i];
    }
    for (int i = 0; i < total; i++)
      if (data[i] != tt.data[i]) return false;
    return true;
  }
  TensorTemp& operator=(const TensorTemp& tt)
  {
    numDim = tt.numDim;
    int total = 1;
    for (int i = 0; i < numDim; i++) {
      dim[i] = tt.dim[i];
      total *= dim[i];
    }
    for (int i = 0; i < total; i++)
      data[i] = tt.data[i];
    name = tt.name;
    return *this;
  }
};

struct TensorTempList {
  int numTensor;
  TensorTemp tensors[MAX_NUM_TENSORS];
  bool operator==(const TensorTempList& ttl) const
  {
    if (numTensor != ttl.numTensor) return false;
    for (int i = 0; i < numTensor; i++)
      if (!(tensors[i] == ttl.tensors[i])) return false;
    return true;
  }
};

class OpTemp {
public:
  OpTemp(int _inputs, int _outputs, OpBase::OpType _type)
  : numInputs(_inputs), numOutputs(_outputs), type(_type) {}
  virtual bool compute(int n, TensorTemp* inputs) = 0;
  virtual bool compute(const TensorTemp& x, const TensorTemp& y) = 0;
public:
  OpBase::OpType type;
  int numInputs, numOutputs;
  TensorTemp outputs[MAX_NUM_OUTPUTS];
};

class MatmulTemp : public OpTemp {
public:
  MatmulTemp(OpBase::ActiMode _mode)
  : OpTemp(2, 1, OpBase::OP_MATMUL), mode(_mode)
  {}

  bool compute(int n, TensorTemp* inputs)
  {
    assert(false);
    return false;
  }

  bool compute(const TensorTemp& input, const TensorTemp& weight)
  {
    if (input.numDim != 2 || weight.numDim != 2) return false;
    if (input.dim[0] != BATCH_SIZE) return false;
    if (input.dim[1] != weight.dim[1]) return false;
    if (weight.dim[0] == BATCH_SIZE) return false;
    outputs[0].numDim = 2;
    outputs[0].dim[0] = BATCH_SIZE;
    outputs[0].dim[1] = weight.dim[0];
    outputs[0].name = "matmul{input[" + input.name
                      + "] weight[" + weight.name + "]}";
    int outputN = outputs[0].dim[0];
    int outputC = outputs[0].dim[1];
    int inputC = input.dim[1];
    for (int i = 0; i < outputN; i++)
      for (int j = 0; j < outputC; j++) {
        GENTYPE val = 0;
        for (int k = 0; k < inputC; k++)
          val += input.data[i * inputC + k]
                 * weight.data[j * inputC + k];
        outputs[0].data[i * outputC + j] = val;
      }
    if (mode == OpBase::AC_MODE_RELU) {
      for (int i = 0; i < outputN * outputC; i++)
        outputs[0].data[i] = std::abs(outputs[0].data[i]);
    } else if (mode == OpBase::AC_MODE_SIGMOID) {
      assert(false);
    } else if (mode == OpBase::AC_MODE_TANH) {
      assert(false);
    } else {
      assert(mode == OpBase::AC_MODE_NONE);
    }
    return true;
  }
private:
  OpBase::ActiMode mode;
};

class ElementTemp : public OpTemp {
public:
  ElementTemp(OpBase::OpType _type)
  : OpTemp(2, 1, _type) {
    assert(_type == OpBase::OP_EW_ADD || _type == OpBase::OP_EW_MUL);
  }

  bool compute(int n, TensorTemp* inputs)
  {
    assert(false);
    return false;
  }

  bool compute(const TensorTemp& x1, const TensorTemp& x2)
  {
    if (x1.numDim != x2.numDim) return false;
    int numDim = x1.numDim;
    int total = 1;
    for (int i = 0; i < numDim; i++) {
      if (x1.dim[i] != x2.dim[i])
        return false;
      total *= x1.dim[i];
    }
    outputs[0].numDim = numDim;
    for (int i = 0; i < numDim; i++)
      outputs[0].dim[i] = x1.dim[i];
    if (type == OpBase::OP_EW_ADD) {
      outputs[0].name = "ewAdd{input[" + x1.name
                        + "] input[" + x2.name + "]}";
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = x1.data[i] + x2.data[i];
    } else {
      assert(type == OpBase::OP_EW_MUL);
      outputs[0].name = "ewMul{input[" + x1.name
                        + "] input[" + x2.name + "]}";
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = x1.data[i] * x2.data[i];
    }

    return true;
  }
};

namespace std {
  template <>
  struct hash<TensorTemp>
  {
    size_t operator()(const TensorTemp& tt) const
    {
      size_t res = 17;
      int total = 1;
      res = res * 31 + hash<int>()(tt.numDim);
      for (int i = 0; i < tt.numDim; i++) {
        res = res * 31 + hash<int>()(tt.dim[i]);
        total *= tt.dim[i];
      }
      for (int i = 0; i < total; i++)
        res = res * 31 + hash<GENTYPE>()(tt.data[i]);
      return res;
    }
  };

  template <>
  struct hash<TensorTempList>
  {
    size_t operator()(const TensorTempList& ttl) const
    {
      size_t res = 17;
      res = res * 31 + hash<int>()(ttl.numTensor);
      for (int i = 0; i < ttl.numTensor; i++)
        res = res * 31 + hash<TensorTemp>()(ttl.tensors[i]);
      return res;
    }
  };
}

void init_tensor_temp(int n, int c, TensorTemp& x1, std::string name);

void dfs(int depth,
         std::vector<TensorTemp>& inputs,
         const std::vector<OpTemp*>& ops,
         std::unordered_map<TensorTempList, std::string>& graphs,
         std::unordered_map<std::string, bool>& transfers);

int main(int argc, char **argv)
{
  std::unordered_map<TensorTempList, std::string> graphs;
  std::vector<TensorTemp> inputs;
  std::unordered_map<std::string, bool> transfers;
  TensorTemp x1, x2, x3, w1, w2, w3;
  init_tensor_temp(BATCH_SIZE, 4, x1, "x1");
  inputs.push_back(x1);
  init_tensor_temp(BATCH_SIZE, 4, x2, "x2");
  inputs.push_back(x2);
  init_tensor_temp(BATCH_SIZE, 4, x3, "x3");
  inputs.push_back(x3);
  init_tensor_temp(4, 4, w1, "w1");
  inputs.push_back(w1);
  init_tensor_temp(4, 4, w2, "w2");
  inputs.push_back(w2);
  init_tensor_temp(4, 4, w3, "w3");
  inputs.push_back(w3);
  std::vector<OpTemp*> ops;
  ops.push_back(new MatmulTemp(OpBase::AC_MODE_NONE));
  //ops.push_back(new MatmulTemp(OpBase::AC_MODE_RELU));
  ops.push_back(new ElementTemp(OpBase::OP_EW_ADD));
  ops.push_back(new ElementTemp(OpBase::OP_EW_MUL));
  dfs(0, inputs, ops, graphs, transfers);
  return 0;
}

void init_tensor_temp(int n, int c, TensorTemp& tt, std::string name)
{
  tt.numDim = 2;
  tt.dim[0] = n;
  tt.dim[1] = c;
  tt.name = name;
  int total = n * c;
  for (int i = 0; i < total; i++)
    tt.data[i] = std::rand();
}

bool find_common_subgraph(const std::string& g1,
                          const std::string& g2,
                          const std::string& prefix)
{
  size_t pos1 = g1.find(prefix, 0);
  while (pos1 != std::string::npos) {
    size_t pos2 = g2.find(prefix, 0);
    while (pos2 != std::string::npos) {
      size_t len = prefix.length();
      while (pos1+len < g1.length() && pos2+len < g2.length() && g1[pos1+len] == g2[pos2+len])
        len ++;
      if (len > 20) return true;
      pos2 = g2.find(prefix, pos2 + 1);
    }
    pos1 = g1.find(prefix, pos1 + 1);
  }
  return false;
}

bool pass_checks(const std::string& g1,
                 const std::string& g2,
                 std::unordered_map<std::string, bool>& transfers)
{
  // Pruning: cannot have common subgraphs
  if (find_common_subgraph(g1, g2, "input")) return false;
  if (find_common_subgraph(g1, g2, "weight")) return false;
  printf("pass check#1\n");
  // Pruning: variable renaming (e.g., "x1" must appear before "x2")
  // Can only apply this pruning to g1
  if (g1.find("x1") > g1.find("x2")) return false;
  if (g1.find("x2") > g1.find("x3")) return false;
  if (g1.find("w1") > g1.find("w2")) return false;
  if (g1.find("w2") > g1.find("w3")) return false;
  printf("pass check#2\n");
  // Pruning: variable substitutions (e.g., replace "w3" with "w2")
  // Pruning: cannot have common supergraphs

  return true;
}

void dfs(int depth,
         std::vector<TensorTemp>& inputs,
         const std::vector<OpTemp*>& ops,
         std::unordered_map<TensorTempList, std::string>& graphs,
         std::unordered_map<std::string, bool>& transfers)
{
  // Add current subgraph to graphs
  std::string name;
  TensorTempList ttl;
  ttl.numTensor = 0;
  for (int i = inputs.size() - 1; i > 5; i--) {
    bool found = false;
    for (int j = i + 1; j < inputs.size(); j++)
      if (inputs[j].name.find(inputs[i].name) != std::string::npos)
        found = true;
    if (!found) {
      ttl.numTensor++;
      assert(ttl.numTensor <= MAX_NUM_TENSORS);
      ttl.tensors[ttl.numTensor-1] =  inputs[i];
      name = name + inputs[i].name + " | ";
    }
  }
  printf("Name[%d]: %s\n", depth, name.c_str());
  if (graphs.find(ttl) != graphs.end()) {
    // Found a match
    std::string oldmatch = graphs[ttl];
    printf("oldmatch = %s\n", oldmatch.c_str());
    if (oldmatch != name) {
      if (pass_checks(oldmatch, name, transfers)) {
        if (transfers.find(oldmatch+name) == transfers.end()) {
          transfers[oldmatch+name] = true;
          printf("Source Graph: %s\n", oldmatch.c_str());
          printf("Target Graph: %s\n", name.c_str());
        }
      }
    }
  } else {
    graphs[ttl] = name;
  }
  if (depth >= 2) return; // MAX NUM OPERATORS
  for (int i = 0; i < ops.size(); i++)
    switch (ops[i]->type) {
      case OpBase::OP_MATMUL:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          for (int k = 0; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k])) {
              inputs.push_back(op->outputs[0]);
              dfs(depth + 1, inputs, ops, graphs, transfers);
              inputs.pop_back();
            }
        break;
      }
      case OpBase::OP_EW_ADD:
      case OpBase::OP_EW_MUL:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          for (int k = 0; k < inputs.size(); k++)
            if (inputs[j].name < inputs[k].name) {
              if (op->compute(inputs[j], inputs[k])) {
                inputs.push_back(op->outputs[0]);
                dfs(depth + 1, inputs, ops, graphs, transfers);
                inputs.pop_back();
              }
            }
        break;
      }
      default:
        assert(false);
    }
}

