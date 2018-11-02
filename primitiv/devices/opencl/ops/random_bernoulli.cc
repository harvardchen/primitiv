#include <primitiv/config.h>

#include <primitiv/devices/opencl/device.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::random_bernoulli_impl(float p, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      state_->queue.enqueueMapBuffer(
        MDATA(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  state_->randomizer_.fill_bernoulli(p, size, mapped_ptr);
  state_->queue.enqueueUnmapMemObject(MDATA(y), mapped_ptr);
}

}  // namespace devices
}  // namespace primitiv
