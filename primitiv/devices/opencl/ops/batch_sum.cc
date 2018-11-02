#include <primitiv/config.h>

#include <primitiv/devices/opencl/device.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t batch = x.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->batch_sum_fw_group_size);
  state_->batch_sum_fw_kernel.setArg(0, CDATA(x));
  state_->batch_sum_fw_kernel.setArg(1, size);
  state_->batch_sum_fw_kernel.setArg(2, batch);
  state_->batch_sum_fw_kernel.setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_sum_fw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_sum_fw_group_size),
      cl::NDRange(state_->batch_sum_fw_group_size));
}

void OpenCL::batch_slice_bw_impl(const Tensor &gy, std::uint32_t offset, Tensor &gx) {
  const std::uint32_t volume = gy.shape().volume();
  const std::uint32_t shift = volume * offset;
  const std::uint32_t size = gy.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->batch_slice_bw_group_size);
  state_->batch_slice_bw_kernel.setArg(0, CDATA(gy));
  state_->batch_slice_bw_kernel.setArg(1, size);
  state_->batch_slice_bw_kernel.setArg(2, MDATA(gx));
  state_->batch_slice_bw_kernel.setArg(3, shift);
  state_->queue.enqueueNDRangeKernel(
      state_->batch_slice_bw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_slice_bw_group_size),
      cl::NDRange(state_->batch_slice_bw_group_size));
}

}  // namespace devices
}  // namespace primitiv
