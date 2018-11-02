#include <primitiv/config.h>

#include <algorithm>
#include <iostream>
// #include <random>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/cl2.hpp>

#include <primitiv/core/error.h>
// #include <primitiv/core/memory_pool.h>
#include <primitiv/core/random.h>
#include <primitiv/devices/opencl/device.h>
#include <primitiv/devices/opencl/internal.h>

namespace {

/**
 * Generates source code of all kernel functions.
 * @return Source code of kernel functions.
 */
std::string generate_kernels() {
  return {
    // `kernels.inc` is generated from `kernels.cl`
#include "primitiv/devices/opencl/kernels.inc"
  };
}

/**
 * Returns the list of available platforms.
 * @return List of available cl::Platform.
 */
std::vector<cl::Platform> get_all_platforms() {
  std::vector<cl::Platform> ret;
  cl::Platform::get(&ret);
  return ret;
}

/**
 * Returns the list of available devices on the specified platform.
 * @param platform_id Platform ID.
 * @return List of available cl::Device.
 */
std::vector<cl::Device> get_all_devices(std::uint32_t platform_id) {
  const auto all_pfs = ::get_all_platforms();
  if (platform_id >= all_pfs.size()) {
    PRIMITIV_THROW_ERROR("Invalid platform ID: " << platform_id);
  }
  std::vector<cl::Device> ret;
  all_pfs[platform_id].getDevices(CL_DEVICE_TYPE_ALL, &ret);
  return ret;
}

/**
 * Returns the cl::Device corresponding to the specified IDs.
 * @param platform_id Platform ID.
 * @param device_id Device ID.
 * @return Corresponding cl::Device object.
 */
cl::Device get_device(std::uint32_t platform_id, std::uint32_t device_id) {
  const auto all_devs = ::get_all_devices(platform_id);
  if (device_id >= all_devs.size()) {
    PRIMITIV_THROW_ERROR(
        "Invalid device ID: " << device_id
        << " (on the platform " << platform_id << ")");
  }
  return all_devs[device_id];
}

}  // namespace

namespace primitiv {
namespace devices {

OpenCLInternalState::OpenCLInternalState(
      std::uint32_t pf_id, std::uint32_t dev_id, std::uint32_t rng_seed)
    : randomizer_(rng_seed)
    , device(::get_device(pf_id, dev_id))
    , context({ device })
    , queue(context, device, 0)
    , pool(
        [this](std::size_t size) -> void * {  // allocator
          return static_cast<void *>(
              new cl::Buffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                size,
                nullptr));
        },
        [this](void *ptr) -> void {  // deleter
          // NOTE(odashi):
          // Deleting cl::Buffer does NOT block the process regardless whether
          // the remaining kernel functions are still working or not.
          // We have to manually wait for finishing all kernel functions to
          // prevent memory corruption.
          queue.finish();
          // Then, we can delete the buffer safely.
          delete static_cast<cl::Buffer *>(ptr);
        }) {
  cl::Program program(context, ::generate_kernels());
  try {
    program.build({device});
  } catch (...) {
    PRIMITIV_THROW_ERROR("OpenCL kernel compile error:" << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
  }

#define CONFIGURE_KERNEL(name) \
  { \
    name##_kernel = cl::Kernel(program, #name "_kernel"); \
    name##_group_size = get_work_group_size(name##_kernel); \
  }

#define CONFIGURE_KERNEL_LIST(name) \
  { \
    for (std::uint32_t i = 0; i <= 10; ++i) { \
      std::ostringstream ss; \
      ss << #name "_kernel_" << (1 << i); \
      name##_kernel[i] = cl::Kernel(program, ss.str().c_str()); \
    } \
    name##_group_size = get_work_group_size(name##_kernel[0]); \
  }

  CONFIGURE_KERNEL_LIST(argmax);
  CONFIGURE_KERNEL_LIST(argmin);
  argmax_group_size = calc_dim1_size(argmax_group_size);
  argmin_group_size = calc_dim1_size(argmin_group_size);

  CONFIGURE_KERNEL(set_identity);

  CONFIGURE_KERNEL(pick_fw);
  CONFIGURE_KERNEL(slice_fw);
  CONFIGURE_KERNEL(concat_fw);

  CONFIGURE_KERNEL(pick_bw);
  CONFIGURE_KERNEL(slice_bw);

  CONFIGURE_KERNEL(negate_fw);
  CONFIGURE_KERNEL(abs_fw);
  CONFIGURE_KERNEL(sqrt_fw);
  CONFIGURE_KERNEL(exp_fw);
  CONFIGURE_KERNEL(log_fw);
  CONFIGURE_KERNEL(tanh_fw);
  CONFIGURE_KERNEL(sigmoid_fw);
  CONFIGURE_KERNEL(softplus_fw);
  CONFIGURE_KERNEL(sin_fw);
  CONFIGURE_KERNEL(cos_fw);
  CONFIGURE_KERNEL(tan_fw);

  CONFIGURE_KERNEL(abs_bw);
  CONFIGURE_KERNEL(sqrt_bw);
  CONFIGURE_KERNEL(exp_bw);
  CONFIGURE_KERNEL(log_bw);
  CONFIGURE_KERNEL(tanh_bw);
  CONFIGURE_KERNEL(sigmoid_bw);
  CONFIGURE_KERNEL(softplus_bw);
  CONFIGURE_KERNEL(sin_bw);
  CONFIGURE_KERNEL(cos_bw);
  CONFIGURE_KERNEL(tan_bw);

  CONFIGURE_KERNEL(transpose_fw);
  CONFIGURE_KERNEL(transpose_bw);

  calc_dim2_sizes(
      transpose_fw_group_size,
      transpose_fw_group_size_x, transpose_fw_group_size_y);
  calc_dim2_sizes(
      transpose_bw_group_size,
      transpose_bw_group_size_x, transpose_bw_group_size_y);

  CONFIGURE_KERNEL(flip_fw);
  CONFIGURE_KERNEL(flip_bw);

  calc_dim2_sizes(
      flip_fw_group_size,
      flip_fw_group_size_x, flip_fw_group_size_y);
  calc_dim2_sizes(
      flip_bw_group_size,
      flip_bw_group_size_x, flip_bw_group_size_y);

  CONFIGURE_KERNEL(permute_dims_fw);
  CONFIGURE_KERNEL(permute_dims_bw);

  CONFIGURE_KERNEL(add_const_fw);
  CONFIGURE_KERNEL(subtract_const_r_fw);
  CONFIGURE_KERNEL(subtract_const_l_fw);
  CONFIGURE_KERNEL(multiply_const_fw);
  CONFIGURE_KERNEL(divide_const_r_fw);
  CONFIGURE_KERNEL(divide_const_l_fw);
  CONFIGURE_KERNEL(pow_const_r_fw);
  CONFIGURE_KERNEL(pow_const_l_fw);
  CONFIGURE_KERNEL(prelu_fw);
  CONFIGURE_KERNEL(elu_fw);

  CONFIGURE_KERNEL(pown_fw);

  CONFIGURE_KERNEL(add_const_bw);
  CONFIGURE_KERNEL(subtract_const_r_bw);
  CONFIGURE_KERNEL(subtract_const_l_bw);
  CONFIGURE_KERNEL(multiply_const_bw);
  CONFIGURE_KERNEL(divide_const_r_bw);
  CONFIGURE_KERNEL(divide_const_l_bw);
  CONFIGURE_KERNEL(pow_const_r_bw);
  CONFIGURE_KERNEL(pow_const_l_bw);
  CONFIGURE_KERNEL(prelu_bw);
  CONFIGURE_KERNEL(elu_bw);

  CONFIGURE_KERNEL(pown_bw);

  CONFIGURE_KERNEL(add_scalar_fw);
  CONFIGURE_KERNEL(subtract_scalar_r_fw);
  CONFIGURE_KERNEL(subtract_scalar_l_fw);
  CONFIGURE_KERNEL(multiply_scalar_fw);
  CONFIGURE_KERNEL(divide_scalar_r_fw);
  CONFIGURE_KERNEL(divide_scalar_l_fw);
  CONFIGURE_KERNEL(pow_scalar_r_fw);
  CONFIGURE_KERNEL(pow_scalar_l_fw);

  CONFIGURE_KERNEL(add_fw);
  CONFIGURE_KERNEL(subtract_fw);
  CONFIGURE_KERNEL(multiply_fw);
  CONFIGURE_KERNEL(divide_fw);
  CONFIGURE_KERNEL(pow_fw);

  CONFIGURE_KERNEL(add_bw);
  CONFIGURE_KERNEL(subtract_bw);
  CONFIGURE_KERNEL(multiply_bw);
  CONFIGURE_KERNEL(divide_bw);
  CONFIGURE_KERNEL(pow_bw);

  CONFIGURE_KERNEL_LIST(max_fw);
  CONFIGURE_KERNEL_LIST(min_fw);
  CONFIGURE_KERNEL_LIST(max_bw);
  CONFIGURE_KERNEL_LIST(min_bw);
  max_fw_group_size = calc_dim1_size(max_fw_group_size);
  min_fw_group_size = calc_dim1_size(min_fw_group_size);
  max_bw_group_size = calc_dim1_size(max_bw_group_size);
  min_bw_group_size = calc_dim1_size(min_bw_group_size);

  CONFIGURE_KERNEL_LIST(sum_fw);
  CONFIGURE_KERNEL_LIST(logsumexp_fw);
  sum_fw_group_size = calc_dim1_size(sum_fw_group_size);
  logsumexp_fw_group_size = calc_dim1_size(logsumexp_fw_group_size);

  CONFIGURE_KERNEL(broadcast_fw);
  CONFIGURE_KERNEL(batch_pick_fw);
  CONFIGURE_KERNEL(batch_slice_fw);
  CONFIGURE_KERNEL(batch_concat_fw);
  CONFIGURE_KERNEL(batch_sum_fw);

  CONFIGURE_KERNEL(batch_pick_bw);
  CONFIGURE_KERNEL(batch_slice_bw);

  CONFIGURE_KERNEL(inplace_multiply_const);
  CONFIGURE_KERNEL(inplace_add);
  CONFIGURE_KERNEL(inplace_subtract);

#undef CONFIGURE_KERNEL
#undef CONFIGURE_KERNEL_LIST
}

std::uint32_t OpenCL::num_platforms() {
  return ::get_all_platforms().size();
}

std::uint32_t OpenCL::num_devices(std::uint32_t platform_id) {
  return ::get_all_devices(platform_id).size();
}

void OpenCL::assert_support(
    std::uint32_t platform_id, std::uint32_t device_id) {
  const cl::Device dev = ::get_device(platform_id, device_id);

  // Checks whether the device is globally available.
  if (!dev.getInfo<CL_DEVICE_AVAILABLE>()) {
    PRIMITIV_THROW_ERROR(
        "OpenCL Device " << device_id << " on the platform " << platform_id
        << " is not available (CL_DEVICE_AVAILABLE == false).");
  }

  // Checks other minimum requirements.
#define CHECK_REQUIREMENT(name, value) \
  { \
    const auto actual = dev.getInfo<name>(); \
    if (actual < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << ", " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  }
#define CHECK_REQUIREMENT_VECTOR(name, index, value) \
  { \
    const auto actual = dev.getInfo<name>()[index]; \
    if (actual < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << "[" << #index << "], " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  } \

  CHECK_REQUIREMENT(CL_DEVICE_GLOBAL_MEM_SIZE, 1ull * (1ull << 30));
  CHECK_REQUIREMENT(CL_DEVICE_LOCAL_MEM_SIZE, 16ull * (1ull << 10));
  CHECK_REQUIREMENT(CL_DEVICE_MAX_WORK_GROUP_SIZE, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 1, 16);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 2, 1);
  // NOTE(odashi): OpenCL does not support explicit grid sizes.

#undef CHECK_REQUIREMENT
#undef CHECK_REQUIREMENT_VECTOR
}

void OpenCL::initialize() {
  assert_support(pf_id_, dev_id_);
  state_.reset(new OpenCLInternalState(pf_id_, dev_id_, rng_seed_));
}

OpenCL::OpenCL(std::uint32_t platform_id, std::uint32_t device_id)
: pf_id_(platform_id)
, dev_id_(device_id)
, rng_seed_(std::random_device()()) {
  initialize();
}

OpenCL::OpenCL(
    std::uint32_t platform_id, std::uint32_t device_id, std::uint32_t rng_seed)
: pf_id_(platform_id)
, dev_id_(device_id)
, rng_seed_(rng_seed) {
  initialize();
}

OpenCL::~OpenCL() {
  // Nothing to do for now.
}

}  // namespace devices
}  // namespace primitiv
