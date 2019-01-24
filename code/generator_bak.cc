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
typedef int TYPE;
#define MAX_SIZE 256
#define MAX_NUM_OPS 8
#define MAX_NUM_TENSORS 8

struct TensorTemp {
  int numDim, dim[MAX_DIM];
  TYPE data[MAX_SIZE];
  // Do not compare the following metadata for equation checks
  int opIdx, tsIdx;
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
    opIdx = tt.opIdx;
    tsIdx = tt.tsIdx;
    return *this;
  }
  inline TYPE get_value(int n, int c, int h, int w) const
  {
    assert(numDim == 4);
    int offset = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3]
               + h * dim[3] + w;
    assert(offset >= 0 && offset < MAX_SIZE);
    return data[offset];
  }
  inline void set_value(int n, int c, int h, int w, TYPE val)
  {
    assert(numDim == 4);
    int offset = n * dim[1] * dim[2] * dim[3] + c * dim[2] * dim[3]
               + h * dim[3] + w;
    assert(offset >= 0 && offset < MAX_SIZE);
    data[offset] = val;
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
  virtual bool compute(int n, TensorTemp* inputs, int opIdx) = 0;
  virtual bool compute(const TensorTemp& x, int opIdx) = 0;
  virtual bool compute(const TensorTemp& x, const TensorTemp& y, int opIdx) = 0;
public:
  OpBase::OpType type;
  int numInputs, numOutputs;
  TensorTemp outputs[MAX_NUM_OUTPUTS];
};

std::map<int, std::string> variable_names;
std::map<const OpTemp*, std::string> operator_names;

struct GraphTemp {
  struct GraphOp {
    const OpTemp* opTemp;
    int opIdx[MAX_NUM_INPUTS], tsIdx[MAX_NUM_INPUTS];
    bool operator==(const GraphOp& gop) const
    {
      if (opTemp != gop.opTemp) return false;
      for (int i = 0; i < opTemp->numInputs; i++) {
        if ((opIdx[i] != gop.opIdx[i]) || (tsIdx[i] != gop.tsIdx[i]))
          return false;
      }
      return true;
    }
  };
  int numOps;
  GraphOp op[MAX_NUM_OPS];
  void push_op(const OpTemp* opTemp, const TensorTemp& tt0)
  {
    assert(opTemp->numInputs == 1);
    op[numOps].opTemp = opTemp;
    op[numOps].opIdx[0] = tt0.opIdx; op[numOps].tsIdx[0] = tt0.tsIdx;
    numOps ++;
  }
  void push_op(const OpTemp* opTemp, const TensorTemp& tt0, const TensorTemp& tt1)
  {
    assert(opTemp->numInputs == 2);
    op[numOps].opTemp = opTemp;
    op[numOps].opIdx[0] = tt0.opIdx; op[numOps].tsIdx[0] = tt0.tsIdx;
    op[numOps].opIdx[1] = tt1.opIdx; op[numOps].tsIdx[1] = tt1.tsIdx;
    numOps ++;
  }
  void pop_op(void)
  {
    numOps --;
  }
  std::string to_string(void)
  {
    //for (int i = 0; i < numOps; i++)
      //printf("[%d] op(%d) input1(%d %d) input2(%d %d)\n", i, op[i].opTemp->type, op[i].opIdx[0], op[i].tsIdx[0], op[i].opIdx[1], op[i].tsIdx[1]);
    std::string name;
    for (int i = numOps - 1; i >= 0; i--)
      for (int j = op[i].opTemp->numOutputs - 1; j >= 0; j--) {
        bool found = false;
        for (int k = i + 1; k < numOps; k++)
          for (int l = 0; l < op[k].opTemp->numInputs; l++)
            if (op[k].opIdx[l] == i && op[k].tsIdx[l] == j)
              found = true;
        if (!found) {
          name = name + to_string(i, j) + " | ";
        }
      }
    return name;
  }
  std::string to_string(int opIdx, int tsIdx)
  {
    if (opIdx < 0) {
      assert(tsIdx == 0);
      assert(variable_names.find(opIdx) != variable_names.end());
      return variable_names[opIdx];
    } else {
      const OpTemp* opTemp = op[opIdx].opTemp;
      assert(operator_names.find(opTemp) != operator_names.end());
      std::string name = operator_names[opTemp] + "["
                       + std::to_string(tsIdx) + "]{";
      for (int i = 0; i < opTemp->numInputs; i++) {
        name = name + "input" + std::to_string(i) + "("
             + to_string(op[opIdx].opIdx[i], op[opIdx].tsIdx[i]) + ")";
      }
      return name + "}";
    }
  }
  int find(std::string name) const
  {
    int idx = 0;
    for (int i = 0; i < numOps; i++) {
      const OpTemp* opTemp = op[i].opTemp;
      for (int j = 0; j < opTemp->numInputs; j++) {
        if (op[i].opIdx[j] < 0) {
          assert(variable_names.find(op[i].opIdx[j]) != variable_names.end());
          if (variable_names[op[i].opIdx[j]] == name)
            return idx;
        }
        idx ++;
      }
    }
    return idx;
  }
  void print(std::string prefix)
  {
    printf("%s\n", prefix.c_str());
    for (int i = 0; i < numOps; i++) {
      const OpTemp* opTemp = op[i].opTemp;
      printf("[%d]  ", opTemp->type);
      for (int j = 0; j < opTemp->numInputs; j++)
        printf("(%d %d) ", op[i].opIdx[j], op[i].tsIdx[j]);
      printf("\n");
    }
  }
};

class ScalarMulTemp : public OpTemp {
public:
  ScalarMulTemp(void)
  : OpTemp(2, 1, OpBase::OP_SCALARMUL)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, const TensorTemp& scalar, int opIdx)
  {
    if (scalar.numDim != 0) return false;
    outputs[0].numDim = input.numDim;
    int total = 1;
    for (int i = 0; i < input.numDim; i++) {
      outputs[0].dim[i] = input.dim[i];
      total *= input.dim[i];
    }
    for (int i = 0; i < total; i++)
      outputs[0].data[i] = input.data[i] * scalar.data[0];
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;   
  }
};

class Conv2DTemp : public OpTemp {
public:
  Conv2DTemp(int _strideH, int _strideW,
             bool _samePad, bool _relu)
  : OpTemp(2, 1, OpBase::OP_CONV2D), strideH(_strideH), strideW(_strideW),
    samePad(_samePad), relu(_relu)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, const TensorTemp& weight, int opIdx)
  {
    if (input.numDim != 4 || weight.numDim != 4) return false;
    if (input.dim[0] != BATCH_SIZE) return false;
    if (input.dim[1] != weight.dim[1]) return false;
    if (weight.dim[0] == BATCH_SIZE) return false;
    int padT, padL;
    if (samePad) {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = BATCH_SIZE;
      outputs[0].dim[1] = weight.dim[0];
      outputs[0].dim[2] = (input.dim[2] + strideH - 1) / strideH;
      outputs[0].dim[3] = (input.dim[3] + strideW - 1) / strideW;
      int padH = max((outputs[0].dim[2] - 1) * strideH + weight.dim[2]
                     - input.dim[2], 0);
      int padW = max((outputs[0].dim[3] - 1) * strideW + weight.dim[3]
                     - input.dim[3], 0);
      padT = padH / 2;
      padL = padW / 2;
    } else {
      outputs[0].numDim = 4;
      outputs[0].dim[0] = BATCH_SIZE;
      outputs[0].dim[1] = weight.dim[0];
      outputs[0].dim[2] = (input.dim[2] - weight.dim[2]) / strideH + 1;
      outputs[0].dim[3] = (input.dim[3] - weight.dim[3]) / strideW + 1;
      padT = 0;
      padL = 0;
    }
    for (int n = 0; n < outputs[0].dim[0]; n++)
      for (int c = 0; c < outputs[0].dim[1]; c++)
        for (int h = 0; h < outputs[0].dim[2]; h++)
          for (int w = 0; w < outputs[0].dim[3]; w++) {
            TYPE val = 0;
            for (int cin = 0; cin < weight.dim[1]; cin ++)
              for (int kh = 0; kh < weight.dim[2]; kh ++)
                for (int kw = 0; kw < weight.dim[3]; kw ++) {
                  int posH = h * strideH + kh - padT;
                  int posW = w * strideW + kw - padL;
                  assert(posH >= -padT && posH <= input.dim[2] + padT);
                  assert(posW >= -padL && posW <= input.dim[3] + padL);
                  if ((posH >= 0) && (posH < input.dim[2])
                  && (posW >= 0) && (posW < input.dim[3])) {
                    int weightVal = weight.get_value(c, cin, kh, kw);
                    int inputVal = input.get_value(n, cin, posH, posW);
                    val += weightVal * inputVal;
                  }
                }
            if (relu) val = std::abs(val);
            outputs[0].set_value(n, c, h, w, val);
          }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
private:
  int strideH, strideW;
  bool relu, samePad;
};

class MatmulTemp : public OpTemp {
public:
  MatmulTemp(OpBase::ActiMode _mode)
  : OpTemp(2, 1, OpBase::OP_MATMUL), mode(_mode)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& input, const TensorTemp& weight, int opIdx)
  {
    if (input.numDim != 2 || weight.numDim != 2) return false;
    if (input.dim[0] != BATCH_SIZE) return false;
    if (input.dim[1] != weight.dim[1]) return false;
    if (weight.dim[0] == BATCH_SIZE) return false;
    outputs[0].numDim = 2;
    outputs[0].dim[0] = BATCH_SIZE;
    outputs[0].dim[1] = weight.dim[0];
    int outputN = outputs[0].dim[0];
    int outputC = outputs[0].dim[1];
    int inputC = input.dim[1];
    for (int i = 0; i < outputN; i++)
      for (int j = 0; j < outputC; j++) {
        TYPE val = 0;
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
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
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
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
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
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = x1.data[i] + x2.data[i];
    } else {
      assert(type == OpBase::OP_EW_MUL);
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = x1.data[i] * x2.data[i];
    }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
};

class ActivationTemp : public OpTemp {
public:
  ActivationTemp(OpBase::OpType _type)
  : OpTemp(1, 1, _type) {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    outputs[0].numDim = x1.numDim;
    int total = 1;
    for (int i = 0; i < x1.numDim; i++) {
      outputs[0].dim[i] = x1.dim[i];
      total *= x1.dim[i];
    }
    if (type == OpBase::OP_RELU) {
      for (int i = 0; i < total; i++)
        outputs[0].data[i] = std::abs(x1.data[i]);
    } else if (type == OpBase::OP_SIGMOID) {
      assert(false);
    } else {
      assert(false);
    }
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    assert(false);
    return false;
  }
};

class ConcatTemp : public OpTemp {
public:
  ConcatTemp(int n, int _axis)
  : OpTemp(n, 1, OpBase::OP_CONCAT), axis(_axis)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    for (int i = 1; i < n; i++)
      if (inputs[i].numDim != inputs[0].numDim) return false;
    if (axis >= inputs[0].numDim) return false;
    for (int i = 1; i < n; i++)
      for (int j = 0; j < inputs[0].numDim; j++)
        if ((j != axis) && (inputs[0].dim[j] != inputs[i].dim[j]))
          return false;
    outputs[0].numDim = inputs[0].numDim;
    for (int i = 0; i < outputs[0].numDim; i++)
      outputs[0].dim[i] = inputs[0].dim[i];
    for (int i = 1; i < n; i++)
      outputs[0].dim[axis] += inputs[i].dim[axis];
    int total = 1;
    for (int i = 0; i < outputs[0].numDim; i++)
      total *= outputs[0].dim[i];
    if (total > MAX_SIZE) return false;
    int outSize = 1, inSize = 1;
    for (int i = 0; i < axis; i++)
      outSize *= inputs[0].dim[i];
    for (int i = axis + 1; i < inputs[0].numDim; i++)
      inSize *= inputs[0].dim[i];
    int outIdx = 0, inIdxs[MAX_NUM_INPUTS];
    for (int i = 0; i < n; i++)
      inIdxs[i] = 0;
    for (int out = 0; out < outSize; out++)
      for (int i = 0; i < n; i++)
        for (int j = 0; j < inputs[i].dim[axis]; j++)
          for (int in = 0; in < inSize; in++) {
            outputs[0].data[outIdx++] = inputs[i].data[inIdxs[i]++];
          }
    assert(outIdx == total);
    outputs[0].opIdx = opIdx;
    outputs[0].tsIdx = 0;
    return true;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    assert(numInputs == 2);
    TensorTemp xs[2];
    xs[0] = x1;
    xs[1] = x2;
    return compute(2, xs, opIdx);
  }
private:
  int axis;
};

class EqualSplitTemp : public OpTemp {
public:
  EqualSplitTemp(int n, int _axis)
  : OpTemp(1, n, OpBase::OP_SPLIT), axis(_axis)
  {}
  bool compute(int n, TensorTemp* inputs, int opIdx)
  {
    assert(false);
    return false;
  }
  bool compute(const TensorTemp& x1, int opIdx)
  {
    if (x1.dim[axis] % numOutputs != 0) return false;
    for (int i = 0; i < numOutputs; i++) {
      outputs[i].numDim = x1.numDim;
      for (int j = 0; j < x1.numDim; j++)
        outputs[i].dim[j] = x1.dim[j];
      outputs[i].dim[axis] = x1.dim[axis] / numOutputs;
      int outSize = 1;
      int inSize = 1;
      for (int j = 0; j < axis; j++)
        outSize = outSize * outputs[i].dim[j];
      for (int j = axis; j < outputs[i].numDim; j++)
        inSize = inSize * outputs[i].dim[j];
      for (int out = 0; out < outSize; out++) {
        int dstIdx = out * inSize, srcIdx = out * inSize * numOutputs + inSize * i;
        for (int in = 0; in < inSize; in++)
          outputs[i].data[dstIdx++] = x1.data[srcIdx++];
      }
      outputs[i].opIdx = opIdx;
      outputs[i].tsIdx = i;
    }
    return true;
  }
  bool compute(const TensorTemp& x1, const TensorTemp& x2, int opIdx)
  {
    assert(false);
    return false;
  }
private:
  int axis;
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
        res = res * 31 + hash<TYPE>()(tt.data[i]);
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

bool find_same_subgraph(const GraphTemp::GraphOp& o1,
                        const GraphTemp::GraphOp& o2)
{
  if (o1.opTemp != o2.opTemp)
    return false;
  for (int i = 0; i < o1.opTemp->numInputs; i++) {
    if ((o1.opIdx[i] != o2.opIdx[i]) || (o1.tsIdx[i] != o2.tsIdx[i])) return false;
    if (o1.opIdx[i] >= 0) return false;
  }
  return true;
}

bool find_same_supergraph(const GraphTemp::GraphOp& o1,
                          const GraphTemp::GraphOp& o2)
{
  if (o1.opTemp != o2.opTemp)
    return false;
  // Only one input is different
  int diff = 0;
  for (int i = 0; i < o1.opTemp->numInputs; i++) {
    if ((o1.opIdx[i] != o2.opIdx[i]) || (o1.opIdx[i] >= 0))
      diff ++;
  }
  if (diff > 1) return false;
  return true;
}

bool variable_ordering(const GraphTemp& g)
{
  if (g.find("x1") > g.find("x2")) return false;
  if (g.find("x2") > g.find("x3")) return false;
  if (g.find("w1") > g.find("w2")) return false;
  if (g.find("w2") > g.find("w3")) return false;
  if (g.find("i1") > g.find("i2")) return false;
  if (g.find("i2") > g.find("i3")) return false;
  if (g.find("w4") > g.find("w5")) return false;
  if (g.find("w5") > g.find("w6")) return false;
  return true;
}

bool pass_checks(const GraphTemp& g1,
                 const GraphTemp& g2)
{
  // Pruning: cannot have common subgraphs
  for (int i = 0; i < g1.numOps; i++)
    for (int j = 0; j < g2.numOps; j++)
      if (find_same_subgraph(g1.op[i], g2.op[j]))
        return false;
  // Pruning: cannot have common supergraphs
  if (find_same_supergraph(g1.op[g1.numOps-1], g2.op[g2.numOps-1]))
    return false;
  // Pruning: check variable ordering (x1 used before x2 before x3)
  if ((!variable_ordering(g1)) && (!variable_ordering(g2)))
    return false;
  // Pruning: variable renaming (e.g., "x1" must appear before "x2")
  return true;
}

bool same_via_subst(const GraphTemp& g1,
                    const GraphTemp& g2,
                    std::map<int, int>& variable_subst)
{
  if (g1.numOps != g2.numOps) return false;
  for (int i = 0; i < g1.numOps; i++) {
    if (g1.op[i].opTemp != g2.op[i].opTemp) return false;
    for (int j = 0; j < g1.op[i].opTemp->numInputs; j++) {
      if (g1.op[i].tsIdx[j] != g2.op[i].tsIdx[j]) return false;
      int op1 = g1.op[i].opIdx[j];
      int op2 = g2.op[i].opIdx[j];
      if ((op1 >= 0) || (op2 >= 0)) {
        if (op1 != op2) return false;
      } else {
        if (variable_subst.find(op1) == variable_subst.end()) {
          variable_subst[op1] = op2;
        } else {
          if (variable_subst[op1] != op2) return false;
        }
      }
    }
  }
  return true;
}

void dfs(int depth,
         GraphTemp& graph,
         std::vector<TensorTemp>& inputs,
         const std::vector<OpTemp*>& ops,
         std::unordered_map<TensorTempList, GraphTemp>& hashmap,
         std::vector<std::pair<GraphTemp, GraphTemp> >& transfers)
{
  // Pruning should not have duplicated tensors
  for (int i = 0; i < inputs.size(); i++)
    for (int j = i + 1; j < inputs.size(); j++) {
      if (inputs[i] == inputs[j])
        return;
    }
  // Pruning should not have duplicated operators
  for (int i = 0; i < graph.numOps; i++)
    for (int j = i + 1; j < graph.numOps; j++) {
      if (graph.op[i] == graph.op[j])
        return;
    }
  // Add current subgraph to graphs
  TensorTempList ttl;
  ttl.numTensor = 0;
  for (int i = inputs.size() - 1; inputs[i].opIdx >= 0; i--) {
    bool found = false;
    for (int j = 0; j < graph.numOps; j++)
      for (int k = 0; k < graph.op[j].opTemp->numInputs; k++)
        if (graph.op[j].opIdx[k] == inputs[i].opIdx
        && graph.op[j].tsIdx[k] == inputs[i].tsIdx)
          found = true;
    if (!found) {
      ttl.numTensor++;
      assert(ttl.numTensor <= MAX_NUM_TENSORS);
      ttl.tensors[ttl.numTensor-1] = inputs[i];
    }
  }
  if (hashmap.find(ttl) != hashmap.end()) {
    // Found a match
    GraphTemp oldgraph = hashmap[ttl];
    if (pass_checks(oldgraph, graph)) {
      // Pruning: cannot have redundant transfers via variable substitutions
      bool found = false;
      for (int i = 0; i < transfers.size(); i++) {
        // first <-> oldgraph, second <-> graph
        {
          std::map<int, int> variable_subst;
          if (same_via_subst(transfers[i].first, oldgraph, variable_subst)
          && same_via_subst(transfers[i].second, graph, variable_subst)) {
            found = true;
            break;
          }
        }
        // first <-> graph, second <-> oldgraph
        {
          std::map<int, int> variable_subst;
          if (same_via_subst(transfers[i].first, graph, variable_subst)
          && same_via_subst(transfers[i].second, oldgraph, variable_subst)) {
            found = true;
            break;
          }
        }
      }
      if (!found) {
        transfers.push_back(std::make_pair(oldgraph, graph));
        printf("Source Graph: %s\n", oldgraph.to_string().c_str());
        printf("Target Graph: %s\n", graph.to_string().c_str());
      }
    }
  } else {
    hashmap[ttl] = graph;
  }
  if (depth >= 3) return; // MAX_NUM_OPS
  for (int i = 0; i < ops.size(); i++)
    switch (ops[i]->type) {
      case OpBase::OP_EW_ADD:
      case OpBase::OP_EW_MUL:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          for (int k = j + 1; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k], depth)) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        break;
      }
      case OpBase::OP_MATMUL:
      case OpBase::OP_CONV2D:
      case OpBase::OP_SCALARMUL:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          for (int k = 0; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k], depth)) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        break;
      }
      case OpBase::OP_RELU:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          if (op->compute(inputs[j], depth)) {
            inputs.push_back(op->outputs[0]);
            graph.push_op(op, inputs[j]);
            dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
            graph.pop_op();
            inputs.pop_back();
          }
        break;
      }
      case OpBase::OP_CONCAT:
      {
        OpTemp* op = ops[i];
        assert(op->numInputs == 2);
        for (int j = 0; j < inputs.size(); j++)
          for (int k = j + 1; k < inputs.size(); k++)
            if (op->compute(inputs[j], inputs[k], depth)) {
              inputs.push_back(op->outputs[0]);
              graph.push_op(op, inputs[j], inputs[k]);
              dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
              graph.pop_op();
              inputs.pop_back();
            }
        break;
      }
      case OpBase::OP_SPLIT:
      {
        OpTemp* op = ops[i];
        for (int j = 0; j < inputs.size(); j++)
          if (op->compute(inputs[j], depth)) {
            for (int k = 0; k < op->numOutputs; k++)
              inputs.push_back(op->outputs[k]);
            graph.push_op(op, inputs[j]);
            dfs(depth + 1, graph, inputs, ops, hashmap, transfers);
            graph.pop_op();
            for (int k = 0; k < op->numOutputs; k++)
              inputs.pop_back();
          }
        break;
      }
      default:
        assert(false);
    }
}

void init_tensor_temp(TensorTemp& tt, std::string name, int opIdx, int tsIdx, int n = 0, int c = 0, int h = 0, int w = 0)
{
  variable_names[opIdx] = name;
  tt.numDim = 0;
  if (n > 0) { tt.numDim ++; tt.dim[0] = n;}
  if (c > 0) { tt.numDim ++; tt.dim[1] = c;}
  if (h > 0) { tt.numDim ++; tt.dim[2] = h;}
  if (w > 0) { tt.numDim ++; tt.dim[3] = w;}
  tt.opIdx = opIdx;
  tt.tsIdx = tsIdx;
  int total = 1;
  for (int i = 0; i < tt.numDim; i++)
    total *= tt.dim[i];
  assert(total <= MAX_SIZE);
  for (int i = 0; i < total; i++)
    tt.data[i] = std::rand() - RAND_MAX / 2;
}

void init_graph_temp(GraphTemp& graph)
{
  graph.numOps = 0;
}

int main(int argc, char **argv)
{
  std::unordered_map<TensorTempList, GraphTemp> hashmap;
  std::vector<std::pair<GraphTemp, GraphTemp> > transfers;
  std::vector<TensorTemp> inputs;
  GraphTemp graph;
  init_graph_temp(graph);
  // Create 2D tensors
  TensorTemp x1, x2, x3, w1, w2, w3;
  init_tensor_temp(x1, "x1", -1, 0, BATCH_SIZE, 4);
  inputs.push_back(x1);
  init_tensor_temp(x2, "x2", -2, 0, BATCH_SIZE, 4);
  inputs.push_back(x2);
  init_tensor_temp(x3, "x3", -3, 0, BATCH_SIZE, 4);
  inputs.push_back(x3);
  init_tensor_temp(w1, "w1", -4, 0, 4, 4);
  inputs.push_back(w1);
  init_tensor_temp(w2, "w2", -5, 0, 4, 4);
  inputs.push_back(w2);
  init_tensor_temp(w3, "w3", -6, 0, 4, 4);
  inputs.push_back(w3);
  // Create 4D tensors
  TensorTemp i1, i2, i3, w4, w5, w6;
  init_tensor_temp(i1, "i1", -7, 0, BATCH_SIZE, 4, 5, 5);
  inputs.push_back(i1);
  init_tensor_temp(i2, "i2", -8, 0, BATCH_SIZE, 4, 5, 5);
  inputs.push_back(i2);
  init_tensor_temp(i3, "i3", -9, 0, BATCH_SIZE, 4, 5, 5);
  inputs.push_back(i3);
  init_tensor_temp(w4, "w4", -10, 0, 4, 4, 3, 3);
  inputs.push_back(w4);
  init_tensor_temp(w5, "w5", -11, 0, 4, 4, 3, 3);
  inputs.push_back(w5);
  init_tensor_temp(w6, "w6", -12, 0, 4, 4, 3, 3);
  inputs.push_back(w6);
  // Create 0D scalar tensors
  TensorTemp s0;
  init_tensor_temp(s0, "s0", -13, 0);
  inputs.push_back(s0);
  std::vector<OpTemp*> ops;
  ops.push_back(new MatmulTemp(OpBase::AC_MODE_NONE));
  operator_names[ops.back()] = "Matmul";
  //ops.push_back(new ElementTemp(OpBase::OP_EW_ADD));
  //operator_names[ops.back()] = "EWAdd";
  //ops.push_back(new ElementTemp(OpBase::OP_EW_MUL));
  //operator_names[ops.back()] = "EWMul";
  ops.push_back(new Conv2DTemp(1, 1, true, false));
  operator_names[ops.back()] = "Conv3x3S";
  ops.push_back(new Conv2DTemp(1, 1, true, true));
  operator_names[ops.back()] = "Conv3x3SR";
  //ops.push_back(new ScalarMulTemp());
  //operator_names[ops.back()] = "ScalarMul";
  //ops.push_back(new ActivationTemp(OpBase::OP_RELU));
  //operator_names[ops.back()] = "Relu";
  ops.push_back(new ConcatTemp(2/*n*/, 1/*axis*/));
  operator_names[ops.back()] = "Concat_1";
  ops.push_back(new ConcatTemp(2/*n*/, 0/*axis*/));
  operator_names[ops.back()] = "Concat_0";
  ops.push_back(new EqualSplitTemp(2/*n*/, 1/*axis*/));
  operator_names[ops.back()] = "Split_1";
  ops.push_back(new EqualSplitTemp(2/*n*/, 0/*axis*/));
  operator_names[ops.back()] = "Split_0";
  // Test
  OpTemp* concat = new ConcatTemp(2, 0);
  OpTemp* conv2d = new Conv2DTemp(1, 1, true, true);
  assert(concat->compute(i1, i2, 0));
  assert(conv2d->compute(concat->outputs[0], w4, 1));
  // Test
  dfs(0, graph, inputs, ops, hashmap, transfers);
}
