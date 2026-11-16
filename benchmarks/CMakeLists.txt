add_executable(mnist_mlp mnist_mlp.cpp)
target_link_libraries(mnist_mlp PRIVATE flashgrad::flashgrad flashgrad_options)
target_compile_definitions(mnist_mlp PRIVATE FLASHGRAD_MNIST_DIR="${FLASHGRAD_MNIST_DIR}")
