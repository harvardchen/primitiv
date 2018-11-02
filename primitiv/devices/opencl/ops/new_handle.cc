#include <primitiv/config.h>

#include <primitiv/devices/opencl/device.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> OpenCL::new_handle(const Shape &shape) {
  return state_->pool.allocate(sizeof(float) * shape.size());
}

}  // namespace devices
}  // namespace primitiv
