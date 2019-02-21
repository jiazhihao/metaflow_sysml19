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
#ifndef _SUBST_EXAMPLES_H_
#define _SUBST_EXAMPLES_H_
#include "substitution.h"

class FuseConvReluDstOp : public DstOp {
public:
  FuseConvReluDstOp(const SrcOp* conv);
  Op create_operator(Model* model);
};

class FuseMmActiDstOp : public DstOp {
public:
  FuseMmActiDstOp(const SrcOp* mm, const SrcOp* acti);
  Op create_operator(Model* model);
};

class MergeMatmulDstOp : public DstOp {
public:
  MergeMatmulDstOp(const SrcOp* conv1, const SrcOp* conv2);
  Op create_operator(Model* model);
};

class MergeConvDstOp : public DstOp {
public:
  MergeConvDstOp(const SrcOp* conv1, const SrcOp* conv2);
  Op create_operator(Model* model);
};

class SameOp : public DstOp {
public:
  SameOp(const SrcOp* op);
  Op create_operator(Model* model);
public:
  Op sameOp;
};

class SplitOp : public DstOp {
public:
  SplitOp(const SrcOp* op1, const SrcOp* op2);
  Op create_operator(Model* model);
};

class NewNoOp : public DstOp {
public:
  NewNoOp(const SrcOp* op);
  Op create_operator(Model* model);
};

class Conv3x3Op : public DstOp {
public:
  Conv3x3Op(const SrcOp* op1);
  Op create_operator(Model* model);
};

class ExConcat : public DstOp {
public:
  ExConcat(const SrcOp* op);
  Op create_operator(Model* model);
};

GraphXfer* create_fuse_conv_batch_xfer(Model* model);
GraphXfer* create_fuse_mm_acti_xfer(Model* model);
GraphXfer* create_fuse_conv_relu_xfer(Model* model);
GraphXfer* create_merge_mm_xfer(Model* model);
GraphXfer* create_merge_conv_xfer(Model* model);
GraphXfer* create_enlarge_conv_xfer(Model* model);
GraphXfer* create_exclusive_concat_xfer(Model* model);
GraphXfer* create_resnet_merge_xfer(Model* model);
#endif
