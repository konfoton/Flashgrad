add_executable(device_info device_info.cpp)
target_link_libraries(device_info PRIVATE flashgrad::flashgrad flashgrad_options)

add_executable(tensor_basics tensor_basics.cpp)
target_link_libraries(tensor_basics PRIVATE flashgrad::flashgrad flashgrad_options)

add_executable(ops_basics ops_basics.cpp)
target_link_libraries(ops_basics PRIVATE flashgrad::flashgrad flashgrad_options)

add_executable(autograd_basics autograd_basics.cpp)
target_link_libraries(autograd_basics PRIVATE flashgrad::flashgrad flashgrad_options)
