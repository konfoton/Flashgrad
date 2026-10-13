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
  core/test_error.cpp
  cuda/test_runtime.cpp
  cuda/test_smoke.cpp
)
target_link_libraries(flashgrad_tests PRIVATE
  flashgrad::flashgrad
  flashgrad_options
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(flashgrad_tests DISCOVERY_MODE PRE_TEST)
