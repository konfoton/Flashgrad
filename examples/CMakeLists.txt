add_executable(device_info device_info.cpp)
target_link_libraries(device_info PRIVATE flashgrad::flashgrad flashgrad_options)

add_executable(tensor_basics tensor_basics.cpp)
target_link_libraries(tensor_basics PRIVATE flashgrad::flashgrad flashgrad_options)

add_executable(ops_basics ops_basics.cpp)
target_link_libraries(ops_basics PRIVATE flashgrad::flashgrad flashgrad_options)

add_executable(autograd_basics autograd_basics.cpp)
target_link_libraries(autograd_basics PRIVATE flashgrad::flashgrad flashgrad_options)

add_executable(dataloader_basics dataloader_basics.cpp)
target_link_libraries(dataloader_basics PRIVATE flashgrad::flashgrad flashgrad_options)
target_compile_definitions(dataloader_basics PRIVATE FLASHGRAD_MNIST_DIR="${FLASHGRAD_MNIST_DIR}")

add_executable(mlp_basics mlp_basics.cpp)
target_link_libraries(mlp_basics PRIVATE flashgrad::flashgrad flashgrad_options)
