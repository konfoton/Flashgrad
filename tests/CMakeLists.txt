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
  autograd/test_grad_elementwise.cpp
  autograd/test_grad_mode.cpp
  autograd/test_gradcheck.cpp
  autograd/test_tensor_autograd.cpp
  core/test_dtype_device.cpp
  core/test_error.cpp
  core/test_shape.cpp
  core/test_tensor.cpp
  core/test_views.cpp
  cuda/test_caching_allocator.cpp
  cuda/test_ops_elementwise.cpp
  cuda/test_ops_fill.cpp
  cuda/test_ops_matmul.cpp
  cuda/test_ops_reduce.cpp
  cuda/test_ops_softmax.cpp
  cuda/test_runtime.cpp
  cuda/test_smoke.cpp
  cuda/test_tensor_cuda.cpp
)
target_include_directories(flashgrad_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(flashgrad_tests PRIVATE
  flashgrad::flashgrad
  flashgrad_options
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(flashgrad_tests DISCOVERY_MODE PRE_TEST)
# The whole suite again with the caching allocator off, so every free goes to cudaFree.
gtest_discover_tests(flashgrad_tests DISCOVERY_MODE PRE_TEST
  TEST_PREFIX "no_caching."
  PROPERTIES ENVIRONMENT "FLASHGRAD_NO_CACHING=1"
)
