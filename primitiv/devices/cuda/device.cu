#include <primitiv/config.h>

#include <random>

#include <primitiv/core/error.h>
#include <primitiv/devices/cuda/device.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

std::uint32_t CUDA::num_devices() {
  int ret;
  CUDA_CALL(::cudaGetDeviceCount(&ret));
  return ret;
}

void CUDA::assert_support(std::uint32_t device_id) {
  if (device_id >= num_devices()) {
    PRIMITIV_THROW_ERROR("Invalid device ID: " << device_id);
  }

  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, device_id));

  // Checks compute capability
  static const int MIN_CC_MAJOR = 3;
  static const int MIN_CC_MINOR = 0;
  if (prop.major < MIN_CC_MAJOR ||
      (prop.major == MIN_CC_MAJOR && prop.minor < MIN_CC_MINOR)) {
    PRIMITIV_THROW_ERROR(
        "CUDA Device " << device_id << " does not satisfy the "
        "minimum requirement of the compute capability: "
        << prop.major << '.' << prop.minor << " < "
        << MIN_CC_MAJOR << '.' << MIN_CC_MINOR);
  }

  // Checks other minimum requirements.
#define CHECK_REQUIREMENT(name, value) \
  { \
    if (prop.name < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "CUDA Device " << device_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << ", " \
          << "value: " << prop.name << ", " \
          << "required at least: " << (value)); \
    } \
  }
#define CHECK_REQUIREMENT_VECTOR(name, index, value) \
  { \
    if (prop.name[index] < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "CUDA Device " << device_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << "[" << #index << "], " \
          << "value: " << prop.name[index] << ", " \
          << "required at least: " << (value)); \
    } \
  }

  CHECK_REQUIREMENT(totalGlobalMem, 1ull * (1ull << 30));
  CHECK_REQUIREMENT(sharedMemPerBlock, 16ull * (1ull << 10));
  CHECK_REQUIREMENT(maxThreadsPerBlock, 256);
  CHECK_REQUIREMENT_VECTOR(maxThreadsDim, 0, 256);
  CHECK_REQUIREMENT_VECTOR(maxThreadsDim, 1, 16);
  CHECK_REQUIREMENT_VECTOR(maxThreadsDim, 2, 1);
  CHECK_REQUIREMENT_VECTOR(maxGridSize, 0, 32767);
  CHECK_REQUIREMENT_VECTOR(maxGridSize, 1, 32767);
  CHECK_REQUIREMENT_VECTOR(maxGridSize, 2, 32767);

#undef CHECK_REQUIREMENT
#undef CHECK_REQUIREMENT_VECTOR
}

void CUDA::initialize() {
  assert_support(dev_id_);

  // Retrieves device properties.
  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, dev_id_));

  // Calculates size of dims to be used in CUDA kernels.
  dim1_x_ = 1;
  while (dim1_x_ < 1024 &&
      dim1_x_ < static_cast<std::uint32_t>(prop.maxThreadsPerBlock)) {
    dim1_x_ <<= 1;
  }
  dim2_y_ = dim1_x_;
  dim2_x_ = 1;
  while (dim2_x_ < dim2_y_) {
    dim2_x_ <<= 1;
    dim2_y_ >>= 1;
  }
  max_batch_ = prop.maxGridSize[1];

  // Initializes additional libraries
  state_.reset(new cuda::InternalState(dev_id_, rng_seed_));
  state_->prop = prop;

  // Initializes the device pointer for integer IDs.
  ids_ptr_ = state_->pool.allocate(sizeof(std::uint32_t) * max_batch_);
}

CUDA::CUDA(std::uint32_t device_id, std::uint32_t rng_seed)
: dev_id_(device_id)
, rng_seed_(rng_seed) {
  initialize();
}

CUDA::CUDA(std::uint32_t device_id)
: CUDA(device_id, std::random_device()()) {}

CUDA::~CUDA() {
  // Nothing to do for now.
}

}  // namespace devices
}  // namespace primitiv
