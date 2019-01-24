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

struct OneOpConstraint {
  OneOpConstraint(Compare _comp, OpBase::OpParameter _para, int _value);
  Compare comp;
  OpBase::OpParameter para;
  int value;
};

class DstOp;
class SrcOp {
public:
  SrcOp(OpBase::OpType _type);
  bool add_constraint(Compare comp, OpBase::OpParameter para, int value);
  bool match(Op op);
public:
  std::vector<OneOpConstraint> constraints;
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

struct SrcEdge {
  SrcEdge(int _idx, SrcOp* _op);
  int idx;
  SrcOp* op;
};

struct SrcEdgeCompare {
  bool operator()(const SrcEdge& a, const SrcEdge& b) const {
    if (a.op != b.op) return a.op < b.op;
    if (a.idx != b.idx) return a.idx < b.idx;
    return false;
  };
};

struct DstEdge {
  DstEdge(int _idx, DstOp* _op);
  int idx;
  DstOp* op;
};

struct DstEdgeCompare {
  bool operator()(const DstEdge& a, const DstEdge& b) const {
    if (a.op != b.op) return a.op < b.op;
    if (a.idx != b.idx) return a.idx < b.idx;
    return false;
  };
};

struct TwoOpConstraint {
  TwoOpConstraint(Compare _comp, SrcOp* src, OpBase::OpParameter srcPara,
                  SrcOp* dst, OpBase::OpParameter dstPara);
  Compare comp;
  SrcOp *op1, *op2;
  OpBase::OpParameter para1, para2;
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
  void add_src_op(SrcOp* op);
  void add_dst_op(DstOp* op);
  void add_src_edge(SrcOp* src, SrcOp* tgt, int idx = 0);
  void add_dst_edge(DstOp* src, DstOp* tgt, int idx = 0);
  bool add_constraint(Compare comp, SrcOp* src, OpBase::OpParameter srcPara,
                      SrcOp* tgt, OpBase::OpParameter dstPara);
  bool map_input(SrcOp* src, DstOp* dst);
  bool map_output(SrcOp* src, DstOp* dst);
  void run(int depth, Graph* graph,
           std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>&,
           std::set<size_t>&, float threshold);
  Graph* create_new_graph(Graph* graph);
public:
  Model* model;
  std::vector<TwoOpConstraint> constraints;
  std::map<SrcOp*, std::set<SrcEdge, SrcEdgeCompare> > srcInEdges, srcOutEdges;
  std::map<DstOp*, std::set<DstEdge, DstEdgeCompare> > dstInEdges, dstOutEdges;
  std::set<Op, OpCompare> mapped;
  std::vector<SrcOp*> srcOps;
  std::vector<DstOp*> dstOps;
};

#endif
