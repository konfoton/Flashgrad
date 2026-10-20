#pragma once

// Whether ops record the autograd graph. On by default; turned off per thread, e.g. for
// inference and parameter updates:
//
//   {
//     flashgrad::NoGradGuard no_grad;
//     w.sub_(w.grad() * lr);  // not recorded
//   }
namespace flashgrad {

bool is_grad_enabled();
void set_grad_enabled(bool enabled);

// Disables recording until the end of the scope, then restores the previous mode.
class NoGradGuard {
 public:
  NoGradGuard() : previous_(is_grad_enabled()) { set_grad_enabled(false); }
  ~NoGradGuard() { set_grad_enabled(previous_); }
  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool previous_;
};

}  // namespace flashgrad
