#include "flashgrad/autograd/grad_mode.h"

namespace flashgrad {

namespace {

thread_local bool grad_enabled = true;

}  // namespace

bool is_grad_enabled() { return grad_enabled; }

void set_grad_enabled(bool enabled) { grad_enabled = enabled; }

}  // namespace flashgrad
