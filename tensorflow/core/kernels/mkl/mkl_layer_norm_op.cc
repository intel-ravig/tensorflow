/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef INTEL_MKL

#include "mkldnn.hpp"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using namespace mkldnn;
using CPUDevice = Eigen::ThreadPoolDevice;

namespace tensorflow {

template <typename Device, typename T>
class MklLayerNormOp : public OpKernel {
 public:
  explicit MklLayerNormOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {
    try {
      const Tensor& src_tensor = MklGetInput(ctx, kSrcIndex);
      const Tensor& scale_tensor = MklGetInput(ctx, kScaleIndex);
      const Tensor& shift_tensor = MklGetInput(ctx, kShiftIndex);

      OP_REQUIRES(ctx, src_tensor.dims() == 2 || src_tensor.dims() == 3,
                  errors::InvalidArgument("input must be 2D or 3D",
                                          src_tensor.shape().DebugString()));

      OP_REQUIRES(ctx, scale_tensor.dims() == 1,
                  errors::InvalidArgument("scale must be 1D tensor",
                                          scale_tensor.shape().DebugString()));
      OP_REQUIRES(ctx, shift_tensor.dims() == 1,
                  errors::InvalidArgument("offset must be 1D tensor",
                                          shift_tensor.shape().DebugString()));
      ctx->set_output(0, src_tensor);

      auto cpu_engine = engine(engine::kind::cpu, 0);
      auto engine_stream = stream(cpu_engine);

      memory::dims src_dims = TFShapeToMklDnnDims(src_tensor.shape());
      memory::dims scale_shift_dims = {2, scale_tensor.dim_size(0)};
      auto src_md =
          memory::desc(src_dims, MklDnnType<T>(),
                       (src_dims.size() == 3) ? memory::format_tag::tnc
                                              : memory::format_tag::nc);
      auto scale_shift_md = memory::desc(scale_shift_dims, MklDnnType<T>(),
                                         memory::format_tag::nc);

      auto lnorm_desc = layer_normalization_forward::desc(
          prop_kind::forward_inference, src_md, epsilon_,
          normalization_flags::use_scale_shift);
      auto lnorm_pd =
          layer_normalization_forward::primitive_desc(lnorm_desc, cpu_engine);
      auto lnorm_prim = layer_normalization_forward(lnorm_pd);

      auto mean_mem = memory(lnorm_pd.mean_desc(), cpu_engine);
      auto variance_mem = memory(lnorm_pd.variance_desc(), cpu_engine);
      void* src_buf =
          static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));
      void* scale_buf =
          static_cast<void*>(const_cast<T*>(scale_tensor.flat<T>().data()));
      void* shift_buf =
          static_cast<void*>(const_cast<T*>(shift_tensor.flat<T>().data()));
      auto src_mem = memory(src_md, cpu_engine, src_buf);
      auto scale_shift_mem = memory(scale_shift_md, cpu_engine);
      void* scale_shift_buf = scale_shift_mem.get_data_handle();
      std::memcpy(scale_shift_buf, scale_buf,
                  sizeof(T) * scale_tensor.dim_size(0));
      std::memcpy(scale_shift_buf + sizeof(T) * scale_tensor.dim_size(0),
                  shift_buf, sizeof(T) * shift_tensor.dim_size(0));

      std::unordered_map<int, memory> lnorm_args;
      lnorm_args.insert({DNNL_ARG_SRC, src_mem});
      lnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
      lnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
      lnorm_args.insert({DNNL_ARG_SCALE_SHIFT, scale_shift_mem});
      lnorm_args.insert({DNNL_ARG_DST, src_mem});
      lnorm_prim.execute(engine_stream, lnorm_args);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 private:
  float epsilon_;
  const int kSrcIndex = 0;
  const int kScaleIndex = 1;
  const int kShiftIndex = 2;
};

REGISTER_KERNEL_BUILDER(
    Name("_MklLayerNorm").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    MklLayerNormOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("_MklLayerNorm").Device(DEVICE_CPU).TypeConstraint<bfloat16>("T"),
    MklLayerNormOp<CPUDevice, bfloat16>);

}  // namespace tensorflow

#endif  // INTEL_MKL
