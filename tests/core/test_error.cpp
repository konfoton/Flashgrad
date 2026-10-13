#include "flashgrad/core/error.h"

#include <gtest/gtest.h>

#include <string>

namespace {

TEST(Error, CheckPassesWhenConditionTrue) { EXPECT_NO_THROW(FG_CHECK(1 + 1 == 2, "math")); }

TEST(Error, CheckThrowsWithFormattedMessage) {
  const int dim = 3;
  try {
    FG_CHECK(dim == 2, "expected 2D tensor, got ", dim, "D");
    FAIL() << "FG_CHECK did not throw";
  } catch (const flashgrad::Error& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("dim == 2"), std::string::npos) << what;
    EXPECT_NE(what.find("expected 2D tensor, got 3D"), std::string::npos) << what;
    EXPECT_NE(what.find("test_error.cpp"), std::string::npos) << what;
    EXPECT_GT(e.line(), 0);
  }
}

TEST(Error, CheckWithoutMessage) {
  try {
    FG_CHECK(false);
    FAIL() << "FG_CHECK did not throw";
  } catch (const flashgrad::Error& e) {
    EXPECT_NE(std::string(e.what()).find("Check failed: false"), std::string::npos);
  }
}

TEST(Error, ThrowIsCatchableAsStdException) {
  EXPECT_THROW(FG_THROW("boom ", 42), std::runtime_error);
  try {
    FG_THROW("boom ", 42);
  } catch (const flashgrad::Error& e) {
    EXPECT_EQ(e.message(), "boom 42");
  }
}

TEST(Error, ConditionEvaluatedOnce) {
  int calls = 0;
  auto f = [&] { return ++calls == 1; };
  FG_CHECK(f(), "should pass");
  EXPECT_EQ(calls, 1);
}

}  // namespace
