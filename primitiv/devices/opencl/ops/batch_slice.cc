#include <primitiv/config.h>

#include <primitiv/devices/opencl/device.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::batch_slice_fw_impl(
    const Tensor &x, std::uint32_t offset, Tensor &y) {
  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t shift = volume * offset;
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->batch_slice_fw_group_size);
  state_->batch_slice_fw_kernel.setArg(0, CDATA(x));
  state_->batch_slice_fw_kernel.setArg(1, shift);
  state_->batch_slice_fw_kernel.setArg(2, size);
  state_->batch_slice_fw_kernel.setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_slice_fw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->batch_slice_fw_group_size),
      cl::NDRange(state_->batch_slice_fw_group_size));
}

}  // namespace devices
}  // namespace primitiv
