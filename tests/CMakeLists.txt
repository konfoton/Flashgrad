include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(flashgrad_tests
  core/test_dtype_device.cpp
  core/test_error.cpp
  core/test_ops_elementwise.cpp
  core/test_ops_linalg.cpp
  core/test_ops_reduce.cpp
  core/test_shape.cpp
  core/test_tensor.cpp
  core/test_views.cpp
  cuda/test_runtime.cpp
  cuda/test_smoke.cpp
  cuda/test_tensor_cuda.cpp
)
target_link_libraries(flashgrad_tests PRIVATE
  flashgrad::flashgrad
  flashgrad_options
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(flashgrad_tests DISCOVERY_MODE PRE_TEST)
