/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopReshape final : public PredicateBasedPass {
  explicit EliminateNopReshape()
      : PredicateBasedPass(PassType::Nop, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_reshape";
  }

  static bool getTargetShape(const Node *n, Graph &graph, std::vector<int64_t> &shape) {
    int opset_version = getOpsetVersion(graph);
    if (opset_version < 5 || opset_version > 12)  // only support opsets from 5 to 11 inclusive
      return false;
  
    auto shape_value = n->inputs()[1];
    Tensor shape_t;
    if (shape_value->node()->kind() == kConstant)
      shape_t = shape_value->node()->t(kvalue);
    else
      shape_t = *(graph.getInitializer(shape_value->uniqueName()));
  
    shape = ParseData<int64_t>(&shape_t);
    return true;
  }

  bool patternMatchPredicate(Node *node) override {
    if (node->kind() != kReshape)
      return false;
    if (node->inputs().size() < 2)
      return false;
    const Value *input = node->inputs()[0];
    if (!input->has_sizes())
      return false;
    auto shape_value = node->inputs()[1];
    if (!((shape_value->node()->kind() == kConstant) ||
          (shape_value->node()->kind() == kParam)))
        return false;
    return true;
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    std::vector<int64_t> target_shape;
    bool success = getTargetShape(node, graph, target_shape);
    if (!success)
      return false;
    auto input_shape = node->inputs()[0]->sizes();
    if (input_shape.size() != target_shape.size())
      return false;

    size_t unknown_dims = 0;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      const auto& dim = input_shape[i];
      if (!dim.is_int || dim.is_unknown) {
        unknown_dims += 1;
        continue;
      }
      if (target_shape[i] == -1) {
        unknown_dims += 1;
        continue;
      }
      if (dim.dim != target_shape[i])
        return false;
    }
    if (unknown_dims > 1)
      return false;
    
    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), node->inputs()[0]);
    if (!replacing_success)
      return false;
    
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
