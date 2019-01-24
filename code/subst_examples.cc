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

#include "subst_examples.h"
// ===================================================
// Rule: fuse_conv_batch
// ===================================================
GraphXfer* create_fuse_conv_batch_xfer(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  SrcOp* conv1 = new SrcOp(OpBase::OP_CONV2D);
  conv1->add_constraint(COMPARE_EQ, OpBase::PM_RELU, false);
  SrcOp* bn1 = new SrcOp(OpBase::OP_BATCHNORM);
  subst->add_src_op(conv1);
  subst->add_src_op(bn1);
  subst->add_src_edge(conv1, bn1);
  DstOp* conv2 = new SameOp(conv1);
  subst->add_dst_op(conv2);
  subst->map_input(conv1, conv2);
  subst->map_output(bn1, conv2);
  return subst;
}

// ===================================================
// Relu: fuse_conv_relu
// ===================================================
FuseConvReluDstOp::FuseConvReluDstOp(const SrcOp* _conv1)
: DstOp(OpBase::OP_CONV2D, _conv1)
{}

Op FuseConvReluDstOp::create_operator(Model* model)
{
  assert(srcOps[0]->type == OpBase::OP_CONV2D);
  assert(srcOps[0]->mapOp.ptr != NULL);
  Conv2D* conv = (Conv2D*) srcOps[0]->mapOp.ptr;
  Tensor input = conv->inputs[0];
  int outputC = conv->outputC;
  int kernelH = conv->kernelH;
  int kernelW = conv->kernelW;
  int strideH = conv->strideH;
  int strideW = conv->strideW;
  int padH = conv->padH;
  int padW = conv->padW;
  bool relu = true;
  Op newConv = model->get_or_create_conv2d(input, outputC, kernelH,
                   kernelW, strideH, strideW, padH, padW, relu);
  return newConv;
}

GraphXfer* create_fuse_conv_relu_xfer(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  SrcOp* conv1 = new SrcOp(OpBase::OP_CONV2D);
  SrcOp* relu1 = new SrcOp(OpBase::OP_RELU);
  subst->add_src_op(conv1);
  subst->add_src_op(relu1);
  subst->add_src_edge(conv1, relu1);
  DstOp* conv2 = new FuseConvReluDstOp(conv1);
  subst->add_dst_op(conv2);
  subst->map_input(conv1, conv2);
  subst->map_output(relu1, conv2);
  return subst;
}

MergeConvDstOp::MergeConvDstOp(const SrcOp* _conv1, const SrcOp* _conv2)
: DstOp(OpBase::OP_CONV2D, _conv1, _conv2)
{}

Op MergeConvDstOp::create_operator(Model* model)
{
  assert(srcOps[0]->type == OpBase::OP_CONV2D);
  assert(srcOps[1]->type == OpBase::OP_CONV2D);
  assert(srcOps[0]->mapOp.ptr != NULL);
  assert(srcOps[1]->mapOp.ptr != NULL);
  Conv2D* conv1 = (Conv2D*) srcOps[0]->mapOp.ptr;
  Conv2D* conv2 = (Conv2D*) srcOps[1]->mapOp.ptr;
  //assert(conv1->kernelH == conv2->kernelH);
  //assert(conv1->kernelW == conv2->kernelW);
  assert(conv1->strideH == conv2->strideH);
  assert(conv1->strideW == conv2->strideW);
  //assert(conv1->padH == conv2->padH);
  //assert(conv1->padW == conv2->padW);
  assert(conv1->relu == conv2->relu);
  Tensor input = conv1->inputs[0];
  int outputC = conv1->outputC + conv2->outputC;
  int kernelH = max(conv1->kernelH, conv2->kernelH);
  int kernelW = max(conv1->kernelW, conv2->kernelW);
  int strideH = conv1->strideH;
  int strideW = conv1->strideW;
  int padH = max(conv1->padH, conv2->padH);
  int padW = max(conv1->padW, conv2->padW);
  bool relu = conv1->relu;
  for (int i = 2; i < conv1->outputs[0].numDim; i++)
    assert(conv1->outputs[0].dim[i] == conv2->outputs[0].dim[i]);

  Op newConv = model->get_or_create_conv2d(input, outputC, kernelH,
                   kernelW, strideH, strideW, padH, padW, relu);
  return newConv;
}

SameOp::SameOp(const SrcOp* op)
: DstOp(op->type, (SrcOp*)op)
{}

Op SameOp::create_operator(Model* model)
{
  return srcOps[0]->mapOp;
}

NewNoOp::NewNoOp(const SrcOp* op)
: DstOp(OpBase::OP_NOOP, (SrcOp*)op)
{}

Op NewNoOp::create_operator(Model* model)
{
  assert(srcOps[0]->mapOp.ptr != NULL);
  OpBase *op = srcOps[0]->mapOp.ptr;
  Op newNoop = model->get_or_create_noop(op->outputs[0]);
  return newNoop;
}

SplitOp::SplitOp(const SrcOp* op1, const SrcOp* op2)
: DstOp(OpBase::OP_SPLIT, op1, op2)
{
}

Op SplitOp::create_operator(Model* model)
{
  assert(srcOps[0]->mapOp.ptr != NULL);
  assert(srcOps[1]->mapOp.ptr != NULL);
  OpBase* op1 = srcOps[0]->mapOp.ptr;
  OpBase* op2 = srcOps[1]->mapOp.ptr;
  assert(op1->outputs[0].numDim == op2->outputs[0].numDim);
  assert(op1->outputs[0].dim[0] == op2->outputs[0].dim[0]);
  for (int i = 2; i < op1->outputs[0].numDim; i++)
    assert(op1->outputs[0].dim[i] == op2->outputs[0].dim[i]);
  int channels[2];
  channels[0] = op1->outputs[0].dim[1];
  channels[1] = op2->outputs[0].dim[1];
  Tensor input = op1->outputs[0];
  input.dim[1] = channels[0] + channels[1];
  Op newSplit = model->get_or_create_split(input, 2, channels);
  return newSplit;
}

GraphXfer* create_merge_conv_xfer(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  SrcOp* op1 = new SrcOp(OpBase::OP_ANY);
  SrcOp* conv1 = new SrcOp(OpBase::OP_CONV2D);
  SrcOp* conv2 = new SrcOp(OpBase::OP_CONV2D);
  subst->add_src_op(op1);
  subst->add_src_op(conv1);
  subst->add_src_op(conv2);
  subst->add_src_edge(op1, conv1);
  subst->add_src_edge(op1, conv2);
  //subst->add_constraint(COMPARE_EQ, conv1, OpBase::PM_KERNEL_H, conv2, OpBase::PM_KERNEL_H);
  //subst->add_constraint(COMPARE_EQ, conv1, OpBase::PM_KERNEL_W, conv2, OpBase::PM_KERNEL_W);
  subst->add_constraint(COMPARE_EQ, conv1, OpBase::PM_STRIDE_H, conv2, OpBase::PM_STRIDE_H);
  subst->add_constraint(COMPARE_EQ, conv1, OpBase::PM_STRIDE_W, conv2, OpBase::PM_STRIDE_W);
  //subst->add_constraint(COMPARE_EQ, conv1, OpBase::PM_PAD_H, conv2, OpBase::PM_PAD_H);
  //subst->add_constraint(COMPARE_EQ, conv1, OpBase::PM_PAD_W, conv2, OpBase::PM_PAD_W);
  subst->add_constraint(COMPARE_EQ, conv1, OpBase::PM_RELU, conv2, OpBase::PM_RELU);
  DstOp* op2 = new SameOp(op1);
  DstOp* conv3 = new MergeConvDstOp(conv1, conv2);
  DstOp* split = new SplitOp(conv1, conv2);
  DstOp* noop1 = new NewNoOp(conv1);
  DstOp* noop2 = new NewNoOp(conv2);
  subst->add_dst_op(op2);
  subst->add_dst_op(conv3);
  subst->add_dst_op(split);
  subst->add_dst_op(noop1);
  subst->add_dst_op(noop2);
  subst->add_dst_edge(op2, conv3);
  subst->add_dst_edge(conv3, split);
  subst->add_dst_edge(split, noop1, 0);
  subst->add_dst_edge(split, noop2, 1);
  subst->map_input(op1, op2);
  subst->map_output(op1, op2);
  subst->map_output(conv1, noop1);
  subst->map_output(conv2, noop2);
  return subst;
}

Conv3x3Op::Conv3x3Op(const SrcOp* op1)
: DstOp(OpBase::OP_CONV2D, op1)
{}

Op Conv3x3Op::create_operator(Model* model)
{
  assert(srcOps[0]->type == OpBase::OP_CONV2D);
  assert(srcOps[0]->mapOp.ptr != NULL);
  Conv2D* conv1 = (Conv2D*) srcOps[0]->mapOp.ptr;
  Op newConv = model->get_or_create_conv2d(conv1->inputs[0],
                        conv1->outputC, 3, 3, conv1->strideH, conv1->strideW,
                        1, 1, conv1->relu);
  return newConv;
}

GraphXfer* create_enlarge_conv_xfer(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  SrcOp* op1 = new SrcOp(OpBase::OP_CONV2D);
  op1->add_constraint(COMPARE_LT, OpBase::PM_KERNEL_H, 4);
  op1->add_constraint(COMPARE_LT, OpBase::PM_KERNEL_W, 4);
  subst->add_src_op(op1);
  DstOp* op2 = new Conv3x3Op(op1);
  subst->add_dst_op(op2);
  subst->map_input(op1, op2);
  subst->map_output(op1, op2);
  return subst;
}

ExConcat::ExConcat(const SrcOp* op)
: DstOp(OpBase::OP_CONCAT, op)
{}

Op ExConcat::create_operator(Model* model)
{
  assert(srcOps[0]->type == OpBase::OP_CONCAT);
  assert(srcOps[0]->mapOp.ptr != NULL);
  Concat* concat = (Concat*) srcOps[0]->mapOp.ptr;
  assert(concat->numInputs == 2);
  bool needCopy[2];
  Tensor inputs[2];
  inputs[0] = concat->inputs[0];
  inputs[1] = concat->inputs[1];
  needCopy[0] = false;
  needCopy[1] = false;
  Op newConcat = model->get_or_create_concat(2, inputs, needCopy);
  return newConcat;
}

GraphXfer* create_exclusive_concat_xfer(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  SrcOp* op1 = new SrcOp(OpBase::OP_ANY);
  op1->add_constraint(COMPARE_NE, OpBase::PM_OP_TYPE, OpBase::OP_SPLIT);
  SrcOp* op2 = new SrcOp(OpBase::OP_ANY);
  op2->add_constraint(COMPARE_NE, OpBase::PM_OP_TYPE, OpBase::OP_SPLIT);
  SrcOp* concat1 = new SrcOp(OpBase::OP_CONCAT);
  concat1->add_constraint(COMPARE_EQ, OpBase::PM_NUM_INPUTS, 2);
  subst->add_src_op(op1);
  subst->add_src_op(op2);
  subst->add_src_op(concat1);
  subst->add_src_edge(op1, concat1);
  subst->add_src_edge(op2, concat1);
  DstOp* op3 = new SameOp(op1);
  DstOp* op4 = new SameOp(op2);
  DstOp* concat2 = new ExConcat(concat1);
  subst->add_dst_op(op3);
  subst->add_dst_op(op4);
  subst->add_dst_op(concat2);
  subst->add_dst_edge(op3, concat2);
  subst->add_dst_edge(op4, concat2);
  subst->map_input(op1, op3);
  subst->map_input(op2, op4);
  subst->map_output(concat1, concat2);
  return subst;
}

GraphXfer* create_resnet_merge_xfer(Model* model)
{
  GraphXfer* subst = new GraphXfer(model);
  SrcOp* conv1 = new SrcOp(OpBase::OP_CONV2D);
  SrcOp* op1 = new SrcOp(OpBase::OP_ANY);
  SrcOp* add1 = new SrcOp(OpBase::OP_EW_ADD);
  subst->add_src_op(conv1);
  subst->add_src_op(op1);
  subst->add_src_op(add1);
  subst->add_src_edge(op1, conv1);
  subst->add_src_edge(op1, add1);
  subst->add_src_edge(conv1, add1);
  DstOp* conv2 = new SameOp(conv1);
  DstOp* op2 = new SameOp(op1);
  subst->add_dst_op(conv2);
  subst->add_dst_op(op2);
  subst->add_dst_edge(op2, conv2);
  subst->map_input(op1, op2);
  subst->map_input(conv1, conv2);
  subst->map_output(op1, op2);
  subst->map_output(add1, conv2);
  return subst;
}

