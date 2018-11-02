#include <primitiv/config.h>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/cl2.hpp>

#include <primitiv/core/memory_pool.h>
#include <primitiv/core/random.h>
#include <primitiv/devices/opencl/device.h>

namespace primitiv {
namespace devices {

/**
 * Hidden objects of OpenCL devices.
 */
struct OpenCLInternalState {
private:
  /**
   * aHelper to obtain maximum work group size of the kernel.
   */
  std::size_t get_work_group_size(const cl::Kernel &kernel) {
    return kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
  }

  /**
   * Helper to find an integer x that satisfy:
   * 1. x == 2^n
   * 2. x <= size
   */
  std::uint32_t calc_dim1_size(std::uint32_t size) {
    std::uint32_t ret = 1;
    while (ret << 1 <= size) ret <<= 1;
    return ret;
  }

  /**
   * Helper to find two sizes (x, y) that satisfy:
   * 1.x == 2^n, y == 2^m
   * 2. x * y <= size
   * 3. x / y == 1 or 2
   */
  void calc_dim2_sizes(std::uint32_t size, std::uint32_t &x, std::uint32_t &y) {
    x = y = 1;
    bool p = true;
    while ((x * y) << 1 <= size) {
      (p ? x : y) <<= 1;
      p = !p;
    }
  }

public:
  OpenCLInternalState(
      std::uint32_t pf_id, std::uint32_t dev_id, std::uint32_t rng_seed);

  DefaultRandomizer randomizer_;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  MemoryPool pool;

#define DECL_KERNEL(name) \
  cl::Kernel name##_kernel; \
  std::uint32_t name##_group_size;
#define DECL_KERNEL_LIST(name, size) \
  std::array<cl::Kernel, size> name##_kernel; \
  std::uint32_t name##_group_size;

  DECL_KERNEL_LIST(argmax, 11);
  DECL_KERNEL_LIST(argmin, 11);

  DECL_KERNEL(set_identity);

  DECL_KERNEL(pick_fw);
  DECL_KERNEL(slice_fw);
  DECL_KERNEL(concat_fw);

  DECL_KERNEL(pick_bw);
  DECL_KERNEL(slice_bw);

  DECL_KERNEL(negate_fw);
  DECL_KERNEL(abs_fw);
  DECL_KERNEL(sqrt_fw);
  DECL_KERNEL(exp_fw);
  DECL_KERNEL(log_fw);
  DECL_KERNEL(tanh_fw);
  DECL_KERNEL(sigmoid_fw);
  DECL_KERNEL(softplus_fw);
  DECL_KERNEL(sin_fw);
  DECL_KERNEL(cos_fw);
  DECL_KERNEL(tan_fw);

  DECL_KERNEL(transpose_fw);
  std::uint32_t transpose_fw_group_size_x;
  std::uint32_t transpose_fw_group_size_y;
  DECL_KERNEL(permute_dims_fw);

  DECL_KERNEL(flip_fw);
  std::uint32_t flip_fw_group_size_x;
  std::uint32_t flip_fw_group_size_y;

  DECL_KERNEL(abs_bw);
  DECL_KERNEL(sqrt_bw);
  DECL_KERNEL(exp_bw);
  DECL_KERNEL(log_bw);
  DECL_KERNEL(tanh_bw);
  DECL_KERNEL(sigmoid_bw);
  DECL_KERNEL(softplus_bw);
  DECL_KERNEL(sin_bw);
  DECL_KERNEL(cos_bw);
  DECL_KERNEL(tan_bw);

  DECL_KERNEL(transpose_bw);
  std::uint32_t transpose_bw_group_size_x;
  std::uint32_t transpose_bw_group_size_y;
  DECL_KERNEL(permute_dims_bw);

  DECL_KERNEL(flip_bw);
  std::uint32_t flip_bw_group_size_x;
  std::uint32_t flip_bw_group_size_y;

  DECL_KERNEL(add_const_fw);
  DECL_KERNEL(subtract_const_r_fw);
  DECL_KERNEL(subtract_const_l_fw);
  DECL_KERNEL(multiply_const_fw);
  DECL_KERNEL(divide_const_r_fw);
  DECL_KERNEL(divide_const_l_fw);
  DECL_KERNEL(pow_const_r_fw);
  DECL_KERNEL(pow_const_l_fw);
  DECL_KERNEL(prelu_fw);
  DECL_KERNEL(elu_fw);

  DECL_KERNEL(pown_fw);

  DECL_KERNEL(add_const_bw);
  DECL_KERNEL(subtract_const_r_bw);
  DECL_KERNEL(subtract_const_l_bw);
  DECL_KERNEL(multiply_const_bw);
  DECL_KERNEL(divide_const_r_bw);
  DECL_KERNEL(divide_const_l_bw);
  DECL_KERNEL(pow_const_r_bw);
  DECL_KERNEL(pow_const_l_bw);
  DECL_KERNEL(prelu_bw);
  DECL_KERNEL(elu_bw);

  DECL_KERNEL(pown_bw);

  DECL_KERNEL(add_scalar_fw);
  DECL_KERNEL(subtract_scalar_r_fw);
  DECL_KERNEL(subtract_scalar_l_fw);
  DECL_KERNEL(multiply_scalar_fw);
  DECL_KERNEL(divide_scalar_r_fw);
  DECL_KERNEL(divide_scalar_l_fw);
  DECL_KERNEL(pow_scalar_r_fw);
  DECL_KERNEL(pow_scalar_l_fw);

  DECL_KERNEL(add_fw);
  DECL_KERNEL(subtract_fw);
  DECL_KERNEL(multiply_fw);
  DECL_KERNEL(divide_fw);
  DECL_KERNEL(pow_fw);

  DECL_KERNEL(add_bw);
  DECL_KERNEL(subtract_bw);
  DECL_KERNEL(multiply_bw);
  DECL_KERNEL(divide_bw);
  DECL_KERNEL(pow_bw);

  DECL_KERNEL_LIST(max_fw, 11);
  DECL_KERNEL_LIST(min_fw, 11);
  DECL_KERNEL_LIST(max_bw, 11);
  DECL_KERNEL_LIST(min_bw, 11);

  DECL_KERNEL_LIST(sum_fw, 11);
  DECL_KERNEL_LIST(logsumexp_fw, 11);

  DECL_KERNEL(broadcast_fw);
  DECL_KERNEL(batch_pick_fw);
  DECL_KERNEL(batch_slice_fw);
  DECL_KERNEL(batch_concat_fw);
  DECL_KERNEL(batch_sum_fw);

  DECL_KERNEL(batch_pick_bw);
  DECL_KERNEL(batch_slice_bw);

  DECL_KERNEL(inplace_multiply_const);
  DECL_KERNEL(inplace_add);
  DECL_KERNEL(inplace_subtract);

#undef DECL_KERNEL
#undef DECL_KERNEL_LIST
};

}  // namespace devices
}  // namespace primitiv
