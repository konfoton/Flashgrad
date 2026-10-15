#include "flashgrad/core/storage.h"

namespace flashgrad {

Storage::Storage(std::size_t nbytes, Device device)
    : impl_(std::make_shared<Impl>(Impl{allocate(nbytes, device), nbytes, device})) {}

void* Storage::data() const noexcept { return impl_ ? impl_->data.get() : nullptr; }

std::size_t Storage::nbytes() const noexcept { return impl_ ? impl_->nbytes : 0; }

Device Storage::device() const noexcept { return impl_ ? impl_->device : Device::cpu(); }

}  // namespace flashgrad
