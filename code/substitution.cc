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

OneOpConstraint::OneOpConstraint(Compare c, OpBase::OpParameter p, int v)
: comp(c), para(p), value(v)
{
}

TwoOpConstraint::TwoOpConstraint(Compare _comp, SrcOp* _o1, OpBase::OpParameter _p1,
                                 SrcOp* _o2, OpBase::OpParameter _p2)
: comp(_comp), op1(_o1), para1(_p1), op2(_o2), para2(_p2)
{
}

SrcOp::SrcOp(OpBase::OpType _type)
: type(_type), mapInput(NULL), mapOutput(NULL)
{}

DstOp::DstOp(OpBase::OpType _type)
: type(_type), mapInput(NULL), mapOutput(NULL)
{}

DstOp::DstOp(OpBase::OpType _type, const SrcOp* op1)
: type(_type), mapInput(NULL), mapOutput(NULL)
{
  srcOps[0] = (SrcOp*) op1;
}

DstOp::DstOp(OpBase::OpType _type, const SrcOp* op1, const SrcOp* op2)
: type(_type), mapInput(NULL), mapOutput(NULL)
{
  srcOps[0] = (SrcOp*) op1;
  srcOps[1] = (SrcOp*) op2;
}

bool SrcOp::add_constraint(Compare comp, OpBase::OpParameter para, int value)
{
  OneOpConstraint ooc(comp, para, value);
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
    OneOpConstraint ooc = constraints[i];
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

SrcEdge::SrcEdge(int _idx, SrcOp* _op)
: idx(_idx), op(_op)
{}

DstEdge::DstEdge(int _idx, DstOp* _op)
: idx(_idx), op(_op)
{}

GraphXfer::GraphXfer(Model* _model)
: model(_model)
{}

void GraphXfer::add_src_op(SrcOp* src)
{
  srcInEdges[src];
  srcOutEdges[src];
  srcOps.push_back(src);
}

void GraphXfer::add_dst_op(DstOp* dst)
{
  dstInEdges[dst];
  dstOutEdges[dst];
  dstOps.push_back(dst);
}

void GraphXfer::add_src_edge(SrcOp* os, SrcOp* od, int idx)
{
  SrcEdge in(idx, os), out(idx, od);
  srcInEdges[od].insert(in);
  srcOutEdges[os].insert(out);
}

void GraphXfer::add_dst_edge(DstOp* os, DstOp* od, int idx)
{
  DstEdge in(idx, os), out(idx, od);
  dstInEdges[od].insert(in);
  dstOutEdges[os].insert(out);
}

bool GraphXfer::add_constraint(Compare comp,
                               SrcOp* src, OpBase::OpParameter srcPara,
                               SrcOp* dst, OpBase::OpParameter dstPara)
{
  TwoOpConstraint gc(comp, src, srcPara, dst, dstPara);
  constraints.push_back(gc);
  return true;
}

bool GraphXfer::map_input(SrcOp* src, DstOp* dst)
{
  assert(src->mapInput == NULL);
  assert(dst->mapInput == NULL);
  src->mapInput = dst;
  dst->mapInput = src;
  return true;
}

bool GraphXfer::map_output(SrcOp* src, DstOp* dst)
{
  assert(src->mapOutput == NULL);
  assert(dst->mapOutput == NULL);
  src->mapOutput = dst;
  dst->mapOutput = src;
  return true;
}

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
      assert(newGraph->check_correctness());
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
        std::set<SrcEdge, SrcEdgeCompare> list = srcInEdges[srcOp];
        std::set<SrcEdge, SrcEdgeCompare>::const_iterator it2;
        // Check edges in the source subgraph
        bool pass = true;
        for (it2 = list.begin(); it2 != list.end(); it2++) {
          SrcEdge edge = *it2;
          if (!graph->has_edge(edge.op->mapOp, op, edge.idx)) pass = false;
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
        if (mapped.find(it->op) != mapped.end()) {
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
          Edge in(it->idx, it->op), out(it->idx, op);
          newGraph->inEdges[op].insert(in);
          newGraph->outEdges[it->op].insert(out);
        }
    } else {
      // Unmapped ops
      std::set<Edge, EdgeCompare> list = opIt->second;
      std::set<Edge, EdgeCompare>::const_iterator it;
      for (it = list.begin(); it != list.end(); it++)
        if (mapped.find(it->op) != mapped.end()) {
          // mapped src -> unmapped dst
          int i = 0;
          for (i = 0; i < srcOps.size(); i++)
            if (srcOps[i]->mapOp.guid == it->op.guid) break;
          assert(i < srcOps.size());
          assert(srcOps[i]->mapOutput != NULL);
          Op op = srcOps[i]->mapOutput->mapOp;
          Edge in(it->idx, op), out(it->idx, opIt->first);
          newGraph->inEdges[opIt->first].insert(in);
          newGraph->outEdges[op].insert(out);
        } else {
          // unmapped src -> unmapped dst
          Edge in(it->idx, it->op), out(it->idx, opIt->first);
          newGraph->inEdges[opIt->first].insert(in);
          newGraph->outEdges[it->op].insert(out);
        }
    }
  // Step 3: add edges in the dstInEdges
  std::map<DstOp*, std::set<DstEdge, DstEdgeCompare> >::iterator dstOpIt;
  for (dstOpIt = dstInEdges.begin(); dstOpIt != dstInEdges.end(); dstOpIt++) {
    std::set<DstEdge, DstEdgeCompare> list = dstOpIt->second;
    std::set<DstEdge, DstEdgeCompare>::const_iterator it;
    for (it = list.begin(); it != list.end(); it++) {
      Op src = it->op->mapOp, dst = dstOpIt->first->mapOp;
      Edge in(it->idx, src), out(it->idx, dst);
      newGraph->inEdges[dst].insert(in);
      newGraph->outEdges[src].insert(out);
    }
  }
  return newGraph;
}

