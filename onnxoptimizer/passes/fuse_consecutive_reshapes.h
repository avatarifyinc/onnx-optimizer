/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveReshapes final : public PredicateBasedPass {
  explicit FuseConsecutiveReshapes()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Partial,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_reshapes";
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() != kReshape)
      return false;

    for (const auto& use: node->outputs()[0]->uses()) {
      if (use.user->kind() != kReshape)
        return false;
    }
    // size_t n_outputs = node->outputs().size();
    // std::cout << "NODE " << node->name() << " outputs: ";
    // for (size_t i = 0; i < n_outputs; i++) {
    //   std::cout << node->outputs()[i]->node()->kind() << " | ";
    //   if (node->outputs()[i]->node()->kind() != kReshape)
    //     return false;
    // }
    // std::cout << std::endl;
    return true;
  }

  bool runTransform(Node* node, Graph&,
                    NodeDestroyType& destroy_current) override {
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
