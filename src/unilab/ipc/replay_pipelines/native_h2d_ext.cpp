#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

void submit_h2d(
    torch::Tensor dst,
    torch::Tensor src,
    std::uintptr_t stream_handle) {
  if (!dst.is_cuda()) {
    throw std::runtime_error("native H2D destination must be a CUDA tensor");
  }
  if (src.is_cuda()) {
    throw std::runtime_error("native H2D source must be a CPU tensor");
  }
  if (!dst.is_contiguous() || !src.is_contiguous()) {
    throw std::runtime_error("native H2D tensors must be contiguous");
  }
  if (dst.numel() != src.numel() || dst.element_size() != src.element_size()) {
    throw std::runtime_error("native H2D source and destination sizes must match");
  }

  auto stream = c10::cuda::CUDAStream::unpack3(
      static_cast<int64_t>(stream_handle),
      dst.device().index(),
      c10::DeviceType::CUDA);
  {
    pybind11::gil_scoped_release release;
    c10::cuda::CUDAStreamGuard guard(stream);
    dst.copy_(src, true);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("submit_h2d", &submit_h2d, "Submit an async CPU-to-CUDA copy on a CUDA stream");
}
