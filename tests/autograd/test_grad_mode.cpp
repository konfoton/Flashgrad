#include <gtest/gtest.h>

#include <thread>

#include "flashgrad/autograd/grad_mode.h"

namespace {

namespace fg = flashgrad;

TEST(GradMode, EnabledByDefault) { EXPECT_TRUE(fg::is_grad_enabled()); }

TEST(GradMode, NoGradGuardRestoresThePreviousMode) {
  {
    const fg::NoGradGuard outer;
    EXPECT_FALSE(fg::is_grad_enabled());
    {
      const fg::NoGradGuard inner;
      EXPECT_FALSE(fg::is_grad_enabled());
    }
    EXPECT_FALSE(fg::is_grad_enabled());  // still inside `outer`
  }
  EXPECT_TRUE(fg::is_grad_enabled());
}

TEST(GradMode, SetGradEnabled) {
  fg::set_grad_enabled(false);
  EXPECT_FALSE(fg::is_grad_enabled());
  {
    // Restores "disabled", not "enabled".
    const fg::NoGradGuard guard;
  }
  EXPECT_FALSE(fg::is_grad_enabled());
  fg::set_grad_enabled(true);
  EXPECT_TRUE(fg::is_grad_enabled());
}

TEST(GradMode, IsPerThread) {
  const fg::NoGradGuard guard;
  bool enabled_in_thread = false;
  std::thread([&] { enabled_in_thread = fg::is_grad_enabled(); }).join();
  EXPECT_TRUE(enabled_in_thread);
  EXPECT_FALSE(fg::is_grad_enabled());
}

}  // namespace
