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

#ifndef _SUBSTITUTION_H_
#define _SUBSTITUTION_H_
#include "ops.h"
#include <queue>

enum Compare {
  COMPARE_EQ,
  COMPARE_NE,
  COMPARE_LT,
  COMPARE_GT
};

struct PMConstraint {
  PMConstraint(Compare comp, PMParameter para, int value);
  Compare comp;
  PMParameter para;
  int value;
};

struct TNConstraint {
  TNConstraint(Compare comp, TNParameter para, DIMParameter dim, int value);
  TNConstraint(Compare comp, TNParameter para1, DIMParameter dim1,
               TNParameter para2, DIMParameter dim2);
  bool singlePara;
  Compare comp;
  TNParameter para1, para2;
  DIMParameter dim1, dim2;
  int value;
};

class OpX;
struct TensorX {
  OpX* op;
  int idx;
};

class OpX {
public:
  OpX(OpBase::OpType _type, TensorX input0, int numOutputs = 1);
  OpX(OpBase::OpType _type, TensorX input0, TensorX input1);
  OpX(OpBase::OpType _type, int n, TensorX* ins);
  bool add_pm_constraint(Compare comp, PMParameter para, int value);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, int);
  bool add_input_constraint(Compare, TNParameter, DIMParameter, TNParameter, DIMParameter);
public:
  OpBase::OpType type;
  Op mapOp;
  int numOutEdges;
  std::vector<TensorX> inputs, outputs;
  std::vector<PMConstraint> pmConstraints;
  std::vector<TNConstraint> tnConstraints;
};

class DstOp;
class SrcOp {
public:
  SrcOp(OpBase::OpType _type);
  bool add_constraint(Compare comp, PMParameter para, int value);
  bool match(Op op);
public:
  std::vector<PMConstraint> constraints;
  OpBase::OpType type;
  Op mapOp;
  DstOp *mapInput, *mapOutput;
};

class DstOp {
public:
  DstOp(OpBase::OpType _type);
  DstOp(OpBase::OpType _type, const SrcOp* op);
  DstOp(OpBase::OpType _type, const SrcOp* op1, const SrcOp* op2);
  virtual Op create_operator(Model* model) = 0;
public:
  OpBase::OpType type;
  Op mapOp;
  SrcOp *mapInput, *mapOutput;
  SrcOp *srcOps[MAX_NUM_INPUTS];
};

template <typename OpType>
struct SubEdge {
  SubEdge(OpType* _srcOp, OpType* _dstOp, int _srcIdx, int _dstIdx)
  : srcOp(_srcOp), dstOp(_dstOp), srcIdx(_srcIdx), dstIdx(_dstIdx) {}
  int srcIdx, dstIdx;
  OpType *srcOp, *dstOp;
};

template<typename OpType>
struct SubEdgeCompare {
  bool operator()(const SubEdge<OpType>& a, const SubEdge<OpType>& b) const {
    if (a.srcOp != b.srcOp) return a.srcOp < b.srcOp;
    if (a.dstOp != b.dstOp) return a.dstOp < b.dstOp;
    if (a.srcIdx != b.srcIdx) return a.srcIdx < b.srcIdx;
    if (a.dstIdx != b.dstIdx) return a.dstIdx < b.dstIdx;
    return false;
  };
};

class GraphCompare {
public:
  bool operator() (Graph* lhs, Graph* rhs) {
    return lhs->total_cost() > rhs->total_cost();
  }
};

class GraphXfer {
public:
  GraphXfer(Model* _model);
  TensorX new_tensor(void);
  bool can_match(OpX* srcOp, Op op, Graph* graph);
  OpX* create_conv2d(TensorX input, TensorX weight,
                     int kernelH, int kernelW,
                     int strideH, int strideW,
                     int padH, int padW,
                     bool relu, bool isSrcOp = true);
  OpX* create_concat(TensorX in1, TensorX in2,
                     int axis, bool isSrcOp = true);
  OpX* create_concat(int n, TensorX* ins,
                     int axis, bool isSrcOp = true);
  OpX* create_split(OpX* concat, TensorX input,
                    int axis, bool isSrcOp = true);
  void add_src_op(SrcOp* op);
  void add_dst_op(DstOp* op);
  void add_src_edge(SrcOp* src, SrcOp* tgt, int srcIdx = 0, int dstIdx = 0);
  void add_dst_edge(DstOp* src, DstOp* tgt, int srcIdx = 0, int dstIdx = 0);
  bool add_constraint(Compare comp, SrcOp* src, PMParameter srcPara,
                      SrcOp* tgt, PMParameter dstPara);
  bool map_input(SrcOp* src, DstOp* dst);
  bool map_output(SrcOp* src, DstOp* dst);
  bool map_output(TensorX src, TensorX dst);
  void run(int depth, Graph* graph,
           std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>&,
           std::set<size_t>&, float threshold);
  Graph* create_new_graph(Graph* graph);
public:
  Model* model;
  int tensorId;
  //std::vector<TwoOpConstraint> constraints;
  //std::map<SrcOp*, std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> > > srcInEdges, srcOutEdges;
  //std::map<DstOp*, std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > > dstInEdges, dstOutEdges;
  std::set<Op, OpCompare> mapped;
  std::map<int, std::pair<Op, int> > mappedTensors;
  std::vector<OpX*> srcOps;
  std::vector<OpX*> dstOps;
};

#endif
