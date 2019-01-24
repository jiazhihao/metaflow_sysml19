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

#include "substitution.h"

// Helper functions
TNParameter to_tn_parameter(bool isInput, int n)
{
  switch (n) {
    case 0: return isInput ? IN_0 : OU_0;
    case 1: return isInput ? IN_1 : OU_1;
    case 2: return isInput ? IN_2 : OU_2;
    case 3: return isInput ? IN_3 : OU_3;
    case 4: return isInput ? IN_4 : OU_4;
    case 5: return isInput ? IN_5 : OU_5;
    default:
      assert(false);
  }
  assert(false);
}

DIMParameter to_dim_parameter(int n)
{
  switch (n) {
    case 0: return DIM_0;
    case 1: return DIM_1;
    case 2: return DIM_2;
    case 3: return DIM_3;
    default:
      assert(false);
  }
  assert(false);
}

PMConstraint::PMConstraint(Compare c, PMParameter p, int v)
: comp(c), para(p), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p, DIMParameter d, int v)
: singlePara(true), comp(c), para1(p), dim1(d), value(v) {}

TNConstraint::TNConstraint(Compare c, TNParameter p1, DIMParameter d1,
                           TNParameter p2, DIMParameter d2)
: singlePara(false), comp(c), para1(p1), dim1(d1), para2(p2), dim2(d2) {}

void add_out_edges(TensorX e)
{
  if (e.op != NULL) e.op->numOutEdges ++;
}

OpX::OpX(OpBase::OpType _type, TensorX in1, int numOutputs)
: type(_type), numOutEdges(0)
{
  inputs.push_back(in1);
  add_out_edges(in1);
  switch (type) {
    case OpBase::OP_SPLIT:
      for (int i = 0; i < numOutputs; i++) {
        TensorX out;
        out.op = this;
        out.idx = i;
        outputs.push_back(out);
      }
      break;
    default:
      assert(false);
  }
}

OpX::OpX(OpBase::OpType _type, TensorX in1, TensorX in2)
: type(_type), numOutEdges(0)
{
  inputs.push_back(in1);
  inputs.push_back(in2);
  add_out_edges(in1);
  add_out_edges(in2);
  TensorX out;
  out.op = this;
  out.idx = 0;
  switch (type) {
    case OpBase::OP_CONV2D:
    case OpBase::OP_CONCAT:
      outputs.push_back(out);
      break;
    default:
      assert(false);
  }
}

OpX::OpX(OpBase::OpType _type, int n, TensorX* ins)
: type(_type), numOutEdges(0)
{
  for (int i = 0; i < n; i++) {
    inputs.push_back(ins[i]);
    add_out_edges(ins[i]);
  }
  TensorX out;
  out.op = this;
  out.idx = 0;
  outputs.push_back(out);
}

bool OpX::add_pm_constraint(Compare comp, PMParameter para, int value)
{
  PMConstraint pmc(comp, para, value);
  pmConstraints.push_back(pmc);
  return true;
}

bool OpX::add_input_constraint(Compare comp, TNParameter para,
                               DIMParameter dim, int value)
{
  TNConstraint tnc(comp, para, dim, value);
  tnConstraints.push_back(tnc);
  return true;
}

bool OpX::add_input_constraint(Compare comp,
                               TNParameter para1, DIMParameter dim1,
                               TNParameter para2, DIMParameter dim2)
{
  TNConstraint tnc(comp, para1, dim1, para2, dim2);
  tnConstraints.push_back(tnc);
  return true;
}

bool SrcOp::add_constraint(Compare comp, PMParameter para, int value)
{
  PMConstraint ooc(comp, para, value);
  constraints.push_back(ooc);
  return true;
}

bool SrcOp::match(Op op)
{
  if (op.guid == 0) return false;
  if (type != OpBase::OP_ANY && type != op.ptr->type)
    return false;
  bool pass = true;
  for (size_t i = 0; i < constraints.size(); i++) {
    PMConstraint ooc = constraints[i];
    int actValue = 0;
    assert(op.ptr->get_parameter(ooc.para, &actValue));
    switch (ooc.comp) {
      case COMPARE_EQ:
        if (actValue != ooc.value) pass = false;
        break;
      case COMPARE_NE:
        if (actValue == ooc.value) pass = false;
        break;
      case COMPARE_LT:
        if (actValue >= ooc.value) pass = false;
        break;
      case COMPARE_GT:
        if (actValue <= ooc.value) pass = false;
        break;
      default:
        assert(false);
    }
  }
  return pass;
}

/*
SrcEdge::SrcEdge(int _idx, SrcOp* _op)
: idx(_idx), op(_op)
{}

DstEdge::DstEdge(int _idx, DstOp* _op)
: idx(_idx), op(_op)
{}
*/

GraphXfer::GraphXfer(Model* _model)
: model(_model), tensorId(10)
{}

OpX* GraphXfer::create_conv2d(TensorX input, TensorX weight,
                              int kernelH, int kernelW,
                              int strideH, int strideW,
                              int padH, int padW,
                              bool relu, bool isSrcOp)
{
  OpX* conv = new OpX(OpBase::OP_CONV2D, input, weight);
  conv->add_pm_constraint(COMPARE_EQ, PM_KERNEL_H, kernelH);
  conv->add_pm_constraint(COMPARE_EQ, PM_KERNEL_W, kernelW);
  conv->add_pm_constraint(COMPARE_EQ, PM_STRIDE_H, strideH);
  conv->add_pm_constraint(COMPARE_EQ, PM_STRIDE_W, strideW);
  conv->add_pm_constraint(COMPARE_EQ, PM_PAD_H, padH);
  conv->add_pm_constraint(COMPARE_EQ, PM_PAD_W, padW);
  conv->add_pm_constraint(COMPARE_EQ, PM_RELU, relu);
  conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_2, kernelH);
  conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_3, kernelW);
  conv->add_input_constraint(COMPARE_EQ, IN_1, DIM_1, IN_0, DIM_1);
  if (isSrcOp)
    srcOps.push_back(conv);
  else
    dstOps.push_back(conv);
  return conv;
}

OpX* GraphXfer::create_concat(TensorX in1, TensorX in2,
                              int axis, bool isSrcOp)
{
  TensorX ins[2];
  ins[0] = in1; ins[1] = in2;
  return create_concat(2, ins, axis, isSrcOp);
}

OpX* GraphXfer::create_concat(int n, TensorX* ins,
                              int axis, bool isSrcOp)
{
  OpX* concat = new OpX(OpBase::OP_CONCAT, n, ins);
  concat->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND, 4);
  for (int i = 1; i < n; i++) {
    TNParameter in_i = to_tn_parameter(true/*is_input*/, i);
    concat->add_input_constraint(COMPARE_EQ, IN_0, DIM_ND,
                                 in_i, DIM_ND);
    for (int j = 0; j < 4; j++) {
      DIMParameter dim_j = to_dim_parameter(j);
      if (j != axis)
        concat->add_input_constraint(COMPARE_EQ, IN_0, dim_j,
                                     in_i, dim_j);
    }
  }
  if (isSrcOp)
    srcOps.push_back(concat);
  else
    dstOps.push_back(concat);
  return concat;
}

OpX* GraphXfer::create_split(OpX* concat, TensorX input,
                             int axis, bool isSrcOp)
{
  OpX* split = new OpX(OpBase::OP_SPLIT, input);
  if (isSrcOp)
    srcOps.push_back(split);
  else
    dstOps.push_back(split);
  return split;
}

TensorX GraphXfer::new_tensor(void)
{
  TensorX t;
  t.op = NULL;
  t.idx = tensorId++;
  return t;
}

bool map_output(TensorX src, TensorX dst)
{
  //TODO to be implemented
  assert(false);
}

//void GraphXfer::add_src_op(SrcOp* src)
//{
//  srcInEdges[src];
//  srcOutEdges[src];
//  srcOps.push_back(src);
//}
//
//void GraphXfer::add_dst_op(DstOp* dst)
//{
//  dstInEdges[dst];
//  dstOutEdges[dst];
//  dstOps.push_back(dst);
//}

//void GraphXfer::add_src_edge(SrcOp* srcOp, SrcOp* dstOp, int srcIdx, int dstIdx)
//{
//  SubEdge<SrcOp> e(srcOp, dstOp, srcIdx, dstIdx);
//  srcInEdges[dstOp].insert(e);
//  srcOutEdges[srcOp].insert(e);
//}

//void GraphXfer::add_dst_edge(DstOp* srcOp, DstOp* dstOp, int srcIdx, int dstIdx)
//{
//  SubEdge<DstOp> e(srcOp, dstOp, srcIdx, dstIdx);
//  dstInEdges[dstOp].insert(e);
//  dstOutEdges[srcOp].insert(e);
//}

//bool GraphXfer::add_constraint(Compare comp,
//                               SrcOp* src, PMParameter srcPara,
//                               SrcOp* dst, PMParameter dstPara)
//{
//  TwoOpConstraint gc(comp, src, srcPara, dst, dstPara);
//  constraints.push_back(gc);
//  return true;
//}

//bool GraphXfer::map_input(SrcOp* src, DstOp* dst)
//{
//  assert(src->mapInput == NULL);
//  assert(dst->mapInput == NULL);
//  src->mapInput = dst;
//  dst->mapInput = src;
//  return true;
//}

//bool GraphXfer::map_output(SrcOp* src, DstOp* dst)
//{
//  assert(src->mapOutput == NULL);
//  assert(dst->mapOutput == NULL);
//  src->mapOutput = dst;
//  dst->mapOutput = src;
//  return true;
//}

bool GraphXfer::can_match(OpX* srcOp, Op op, Graph* graph)
{
  if (srcOp->type != op.ptr->type) return false;
  // check pmConstraints
  for (size_t i = 0; i < srcOp->pmConstraints.size(); i++) {
    PMConstraint pmc = srcOp->pmConstraints[i];
    int actValue = 0;
    assert(op.ptr->get_parameter(pmc.para, &actValue));
    switch (pmc.comp) {
      case COMPARE_EQ:
        if (actValue != pmc.value) return false;
      case COMPARE_NE:
        if (actValue == pmc.value) return false;
      case COMPARE_LT:
        if (actValue >= pmc.value) return false;
      case COMPARE_GT:
        if (actValue <= pmc.value) return false;
      default:
        assert(false);
    }
  }
  // check inputs
  for (size_t i = 0; i < srcOp->inputs.size(); i++) {
    TensorX in = srcOp->inputs[i];
    if (in.op == NULL) {
      // input tensor
      if (mappedTensors.find(in.idx) != mappedTensors.end()) {
        Op mappedOp = mappedTensors[in.idx].first;
        int mappedIdx = mappedTensors[in.idx].second;
        if (!(graph->has_edge(mappedOp, op, mappedIdx, i)))
          return false;
      } else {
        // mapped in.idx to an op
        std::set<Edge, EdgeCompare> list = graph->inEdges.find(op)->second;
        std::set<Edge, EdgeCompare>::const_iterator it2;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          Edge e = *it2;
          if (e.dstIdx == i)
            mappedTensors[in.idx] = std::make_pair(e.srcOp, e.srcIdx);
        }
      }
    } else {
      // intermediate tensor
      assert(in.op->mapOp.ptr != NULL);
      if (!(graph->has_edge(in.op->mapOp, op, in.idx, i)))
        return false;
    }
  }
  // check tnConstraints
  for (size_t i = 0; i < srcOp->tnConstraints.size(); i++) {
    TNConstraint tnc = srcOp->tnConstraints[i];
    int actValue = 0, expValue = 0;
    if (tnc.singlePara) {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      expValue = tnc.value;
    } else {
      assert(op.ptr->get_input_parameter(tnc.para1, tnc.dim1, &actValue));
      assert(op.ptr->get_input_parameter(tnc.para2, tnc.dim2, &expValue));
    }
    switch (tnc.comp) {
      case COMPARE_EQ:
        if (actValue != expValue) return false;
      case COMPARE_NE:
        if (actValue == expValue) return false;
      case COMPARE_LT:
        if (actValue >= expValue) return false;
      case COMPARE_GT:
        if (actValue <= expValue) return false;
      default:
        assert(false);
    }
  }
  return true;
}

void GraphXfer::run(int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::set<size_t>& hashmap, float threshold)
{
  if (depth >= srcOps.size()) {
  } else {
    OpX* srcOp = srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      if (can_match(srcOp, it->first, graph)) {
        Op op = it->first;
        // Check mapOutput
      }
    }
  }
}

/*
void GraphXfer::run(int depth, Graph* graph,
                    std::priority_queue<Graph*, std::vector<Graph*>, GraphCompare>& candidates,
                    std::set<size_t>& hashmap, float threshold)
{
  if (depth >= srcOps.size()) {
    // Check two op constraints
    bool pass = true;
    for (size_t i = 0; i < constraints.size(); i++) {
      TwoOpConstraint toc = constraints[i];
      int value1, value2;
      assert(toc.op1->mapOp.ptr != NULL);
      assert(toc.op2->mapOp.ptr != NULL);
      assert(toc.op1->mapOp.ptr->get_parameter(toc.para1, &value1));
      assert(toc.op2->mapOp.ptr->get_parameter(toc.para2, &value2));
      switch (toc.comp) {
        case COMPARE_EQ:
          if (value1 != value2) pass = false;
          break;
        case COMPARE_NE:
          if (value1 == value2) pass = false;
          break;
        case COMPARE_LT:
          if (value1 >= value2) pass = false;
          break;
        case COMPARE_GT:
          if (value1 <= value2) pass = false;
          break;
        default:
          assert(false);
      }
    }
    // Generate a new graph by applying xfer rule
    if (pass) {
      Graph* newGraph = create_new_graph(graph);
      //assert(newGraph->check_correctness());
      if (newGraph->total_cost() < threshold) {
        if (hashmap.find(newGraph->hash()) == hashmap.end()) {
          hashmap.insert(newGraph->hash());
          candidates.push(newGraph);
        }
      } else {
        delete newGraph;
      }
    }
  } else {
    // Match srcOps[depth];
    SrcOp* srcOp = srcOps[depth];
    std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator it;
    for (it = graph->inEdges.begin(); it != graph->inEdges.end(); it++) {
      if (srcOp->match(it->first)
      && (mapped.find(it->first) == mapped.end())) {
        Op op = it->first;
        std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> > list = srcInEdges[srcOp];
        std::set<SubEdge<SrcOp>, SubEdgeCompare<SrcOp> >::const_iterator it2;
        // Check edges in the source subgraph
        bool pass = true;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          SubEdge<SrcOp> edge = *it2;
          if (!graph->has_edge(edge.srcOp->mapOp, op, edge.srcIdx, edge.dstIdx)) pass = false;
        }
        // Check mapInput/mapOutput
        bool extraInputs = false, extraOutputs = false;
        if (srcInEdges[srcOp].size() != graph->num_in_edges(op))
          extraInputs = true;
        if (srcOutEdges[srcOp].size() != graph->num_out_edges(op))
          extraOutputs = true;
        if (!srcOp->mapInput && extraInputs)
          pass = false;
        if (!srcOp->mapOutput && extraOutputs)
          pass = false;
        // Serch for the next op if pass the check
        if (pass) {
          srcOp->mapOp = op;
          mapped.insert(op);
          run(depth + 1, graph, candidates, hashmap, threshold);
          mapped.erase(op);
          srcOp->mapOp.guid = 0;
          srcOp->mapOp.ptr = NULL;
        }
      }
    }
  }
}

Graph* GraphXfer::create_new_graph(Graph* graph)
{
  Graph* newGraph = new Graph(graph->model);
  // Step 1: add operators to the graph
  std::vector<DstOp*>::iterator dstIt;
  std::map<Op, std::set<Edge, EdgeCompare>, OpCompare>::const_iterator opIt;
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mapped.find(opIt->first) == mapped.end()) {
      newGraph->inEdges[opIt->first];
      newGraph->outEdges[opIt->first];
    }
  for (dstIt = dstOps.begin(); dstIt != dstOps.end(); dstIt ++) {
    DstOp* dstOp = *dstIt;
    dstOp->mapOp = dstOp->create_operator(graph->model);
    newGraph->inEdges[dstOp->mapOp];
    newGraph->outEdges[dstOp->mapOp];
  }
  // Step 2: add edges to the graph
  for (opIt = graph->inEdges.begin(); opIt != graph->inEdges.end(); opIt++)
    if (mapped.find(opIt->first) != mapped.end()) {
      // Mapped ops
      std::set<Edge, EdgeCompare> list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mapped.find(it->srcOp) != mapped.end()) {
          // mapped src -> mapped dst
          // Do nothing!
        } else {
          // unmapped src -> mapped dst
          int i = 0;
          for (i = 0; i < srcOps.size(); i++)
            if (srcOps[i]->mapOp.guid == opIt->first.guid) break;
          assert(i < srcOps.size());
          assert(srcOps[i]->mapInput != NULL);
          Op op = srcOps[i]->mapInput->mapOp;
          Edge e(it->srcOp, op, it->srcIdx, it->dstIdx);
          newGraph->inEdges[op].insert(e);
          newGraph->outEdges[it->srcOp].insert(e);
        }
    } else {
      // Unmapped ops
      std::set<Edge, EdgeCompare> list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mapped.find(it->srcOp) != mapped.end()) {
          // mapped src -> unmapped dst
          int i = 0;
          for (i = 0; i < srcOps.size(); i++)
            if (srcOps[i]->mapOp.guid == it->srcOp.guid) break;
          assert(i < srcOps.size());
          assert(srcOps[i]->mapOutput != NULL);
          Op op = srcOps[i]->mapOutput->mapOp;
          Edge e(op, opIt->first, it->srcIdx, it->dstIdx);
          newGraph->inEdges[opIt->first].insert(e);
          newGraph->outEdges[op].insert(e);
        } else {
          // unmapped src -> unmapped dst
          Edge e(it->srcOp, opIt->first, it->srcIdx, it->dstIdx);
          newGraph->inEdges[opIt->first].insert(e);
          newGraph->outEdges[it->srcOp].insert(e);
        }
    }
  // Step 3: add edges in the dstInEdges
  std::map<DstOp*, std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > >::iterator dstOpIt;
  for (dstOpIt = dstInEdges.begin(); dstOpIt != dstInEdges.end(); dstOpIt++) {
    std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> > list = dstOpIt->second;
    std::set<SubEdge<DstOp>, SubEdgeCompare<DstOp> >::const_iterator it;
    for (it = list.begin(); it != list.end(); it++) {
      Op src = it->srcOp->mapOp, dst = dstOpIt->first->mapOp;
      Edge e(src, dst, it->srcIdx, it->dstIdx);
      newGraph->inEdges[dst].insert(e);
      newGraph->outEdges[src].insert(e);
    }
  }
  return newGraph;
}
*/
