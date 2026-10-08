ggml is pinned at v0.21.0 in `../ggml`. The single
`0001-ram-ggml-v0210-integration.patch` is applied idempotently by CMake before
building. It extends tensor-name capacity for RAM checkpoint names and adds a
diagnostic before the scheduler rejects an unsupported node. CUDA additions
allow contiguous Q8_0 weights with F16 activations through the existing cuBLAS
implementation and honor `GGML_PREC_F32` for FP32 accumulation/output while
retaining FP16 GEMM inputs. The runner requests this precision for tagging;
caption execution retains its separately validated arithmetic.
The complete patch was replayed from pristine v0.21.0, checked for reverse
application, and compared byte-for-byte with the configured source tree.
Do not edit the
generated `third_party/ggml` source tree directly.
