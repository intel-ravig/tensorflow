/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

// This file uses MKL-DNN InnerProduct for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) with bias (BiasAdd) operations.
#ifdef INTEL_MKL

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/kernels/mkl/mkl_quantized_conv_ops.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// MatMul operation does not impose any restriction on which of the
// inputs (first or second) is activation or weight. Typically, first one is
// activation and the second one is weight. Hence, we interpret T1 as
// activation type and T2 as weight type.
template <typename Device, typename T1, typename T2, typename Tbias,
          typename Toutput, bool native_format = false>
class MklFusedMatMulOp : public MklDnnMatMulOpBase<T2, Tbias, Toutput> {
 public:
  explicit MklFusedMatMulOp(OpKernelConstruction* ctx)
      : MklDnnMatMulOpBase<T2, Tbias, Toutput>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("is_filter_const", &(this->is_weight_const_)));

    OP_REQUIRES(ctx, fused_ops_.size() <= 2,
                errors::InvalidArgument(
                    "MklFusedMatMul must have 2 post-arguments at most."));
    OP_REQUIRES(
        ctx, fused_ops_[0] == "BiasAdd",
        errors::InvalidArgument(
            "The 1st post-argument of MklFusedMatMul must be BiasAdd."));
    if (fused_ops_.size() > 1 && fused_ops_[1] == "Add") fuse_add_ = true;
    OP_REQUIRES(
        ctx, transpose_a_ == false,
        errors::InvalidArgument("In[0] of MklMatMul can't be transposed."));
    if (fused_ops_.size() == 2 && fused_ops_[1] == "LeakyRelu") {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("leakyrelu_alpha", &leakyrelu_alpha));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    // FusedMatMul has 3 inputs: src, weights, bias
    const Tensor& src_tensor = ctx->input(this->kInputIndexSrc);
    const Tensor& weight_tensor = ctx->input(this->kInputIndexWeight);
    const Tensor& bias_tensor = MklGetInput(ctx, this->kInputIndexBias);

    MklDnnShape src_mkl_shape;
    MklDnnShape weight_mkl_shape;
    GetMklShape(ctx, this->kInputIndexSrc, &src_mkl_shape, native_format);
    GetMklShape(ctx, this->kInputIndexWeight, &weight_mkl_shape, native_format);
    OP_REQUIRES(ctx, !weight_mkl_shape.IsMklTensor(),
                errors::InvalidArgument("Weight should not be in MKL Layout"));

    // Get shapes of input tensors
    auto src_tf_shape = src_mkl_shape.IsMklTensor() ? src_mkl_shape.GetTfShape()
                                                    : src_tensor.shape();
    auto weight_tf_shape = weight_tensor.shape();

    // Check the constraint of input matrix and bias
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(src_tf_shape),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weight_tf_shape),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias_tensor.shape()),
                errors::InvalidArgument("Biases must be 1D"));

    // Expression: [batch, k] * [k, channel] + [channel] = [batch, channel]
    //
    // Get dimension size of each matrix, dim_pair[] is the location of k
    // in the inputs, we have constraint that k of the two inputs are
    // the same
    const int dim_pair[] = {1, transpose_b_ ? 1 : 0};
    const int batch = src_tf_shape.dim_size(1 - dim_pair[0]);
    const int k = src_tf_shape.dim_size(dim_pair[0]);
    const int channel = weight_tf_shape.dim_size(1 - dim_pair[1]);

    OP_REQUIRES(
        ctx, k == weight_tf_shape.dim_size(dim_pair[1]),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
            ", In[1]: ", weight_tf_shape.DebugString()));
    OP_REQUIRES(ctx, bias_tensor.shape().dim_size(0) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel size: ",
                    bias_tensor.shape().DebugString(), " vs. ", channel));

    // For inputs s[batch, k], w[k, channel] and b[channel], the primitive
    // dims should be described like this:
    //   s[batch, k] * w^T[channel, k] + b[channel] = dst[batch, channel]
    //    [n,    ic] *    [oc,     ic] +  [oc]      =    [n,          oc]
    memory::dims src_dims = memory::dims({batch, k});
    // Reverse the weights dims from [k, channel] to [channel, k].
    memory::dims weight_dims = memory::dims({channel, k});
    memory::dims bias_dims = memory::dims({channel});
    memory::dims dst_dims = memory::dims({batch, channel});
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format =
        transpose_b_ ? memory::format_tag::oi : memory::format_tag::io;

    // Set weight format for primitive:
    //   1. const, let MKL-DNN determine format because it will be cached;
    //   2. var, keep the original format to avoid reordering.
    MklDnnMatMulFwdParams matmul_params(
        src_dims, weight_dims, bias_dims, dst_dims, src_format,
        memory::format_tag::any, memory::format_tag::nc);

    // Extend the basic parameters for data types and fusions.
    ExtendMklDnnMatMulFwdParams(ctx, matmul_params);
    MklDnnMatMulFwdPrimitive<float, T1, T2, Tbias, Toutput>* matmul_prim =
        MklDnnMatMulFwdPrimitiveFactory<float, T1, T2, Tbias, Toutput>::Get(
            matmul_params, false);

    // Allocate output tensor.
    Tensor* dst_tensor = nullptr;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> matmul_pd =
        matmul_prim->GetPrimitiveDesc();

    // The output shape of MatMul is same both for MKL and TF version.
    // They are all NC format, no matter what's the format of input.
    // And the shape of AddOp is also the same with output's shape.
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);

    TensorShape output_tf_shape({batch, channel});

    if (fuse_add_) {
      if (native_format)
        kInputIndex_Add = ctx->num_inputs() - 1;
      else
        kInputIndex_Add = ctx->num_inputs() / 2 - 1;
      const Tensor& add_tensor = MklGetInput(ctx, kInputIndex_Add);
      MklDnnShape add_mkl_shape;
      GetMklShape(ctx, kInputIndex_Add, &add_mkl_shape, native_format);

      // For native format, we need not to set metadata.
      if (native_format && ctx->forward_input_to_output_with_shape(
                               kInputIndex_Add, kOutputIndex_Dst,
                               output_tf_shape, &dst_tensor)) {
        ;  // Need to do nothing for native format
      } else if (!native_format && ForwardMklTensorInToOutWithMklShape(
                                       ctx, kInputIndex_Add, kOutputIndex_Dst,
                                       &dst_tensor, output_mkl_shape, false)) {
        ;  // If it's not native format, need to forward and set meta first
      } else {
        // If forward is not successful, we should use reorder to copy add
        // tensor to dst tensor
        AllocateOutputSetMklShape(ctx, kOutputIndex_Dst, &dst_tensor,
                                  output_tf_shape, output_mkl_shape,
                                  native_format);
        auto output_format_tag =
            MklTensorFormatToMklDnnDataFormat(MklTensorFormat::FORMAT_NC);
        auto add_md = add_mkl_shape.IsMklTensor()
                          ? add_mkl_shape.GetMklLayout()
                          : memory::desc(dst_dims, MklDnnType<Toutput>(),
                                         output_format_tag);
        auto dst_md =
            memory::desc(dst_dims, MklDnnType<Tbias>(), output_format_tag);

        void* add_buf = static_cast<void*>(
            const_cast<Toutput*>(add_tensor.flat<Toutput>().data()));
        void* dst_buf =
            static_cast<void*>((dst_tensor)->flat<Toutput>().data());

        if (native_format) {
          // We are simply deep copying the add_tensor to dst_tensor without
          // changing memory layout, hence using same memory descriptor.
          add_md = dst_md =
              memory::desc({add_tensor.NumElements()}, MklDnnType<Toutput>(),
                           mkldnn::memory::format_tag::x);
        }

        auto fuse_add_src_ = memory(add_md, this->cpu_engine_, add_buf);
        auto fuse_add_dst_ = memory(dst_md, this->cpu_engine_, dst_buf);
        auto reorder_desc =
            ReorderPd(this->cpu_engine_, add_md, this->cpu_engine_, dst_md);

        CreateAndExecuteReorder(reorder_desc, fuse_add_src_, fuse_add_dst_,
                                this->cpu_engine_, ctx);
      }
    } else {
      AllocateOutputSetMklShape(ctx, 0, &dst_tensor, output_tf_shape,
                                output_mkl_shape, native_format);
    }

    // if there's nothing to compute, just return.
    if (batch == 0 || channel == 0) {
      return;
    }

    try {
      // Prepare the input and output for primitive.
      T1* src_data = const_cast<T1*>(src_tensor.flat<T1>().data());
      T2* weight_data = const_cast<T2*>(weight_tensor.flat<T2>().data());
      Tbias* bias_data = const_cast<Tbias*>(bias_tensor.flat<Tbias>().data());
      Toutput* dst_data =
          const_cast<Toutput*>(dst_tensor->flat<Toutput>().data());

      // Reorder input if necessary.
      MklDnnData<T1> src_mkl(&(this->cpu_engine_));
      MklDnnData<T2> weight_mkl(&(this->cpu_engine_));

      auto src_md = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<T1>(), src_format);

      if (src_md != matmul_pd->src_desc()) {
        src_mkl.SetUsrMem(src_md, src_data);
        src_mkl.CheckReorderToOpMem(matmul_pd.get()->src_desc(),
                                    this->cpu_engine_, ctx);
        src_data = reinterpret_cast<T1*>(src_mkl.GetOpMem().get_data_handle());
      }

      // Get cached data when weight is const.
      const memory::desc weight_md =
          memory::desc(weight_dims, MklDnnType<T2>(), weight_format);
      if (weight_md != matmul_pd->weights_desc()) {
        T2* cached_weight_data = nullptr;

        if (this->is_weight_const_) {
          if (this->IsWeightCacheEmpty(ctx)) {
            this->CacheWeight(ctx, matmul_pd, cached_weight_data, weight_tensor,
                              weight_mkl, weight_md);
          }
          cached_weight_data =
              this->GetCachedWeight(ctx, matmul_pd->weights_desc());
        }

        // Cache weight may fail when it gets different format in different
        // iteration. Fallback to reoder if it happens.
        // Also do generel reorder if weight isn't const.
        if (cached_weight_data != nullptr) {
          weight_data = cached_weight_data;
        } else {
          weight_mkl.SetUsrMem(weight_md, weight_data);
          weight_mkl.CheckReorderToOpMem(matmul_pd.get()->weights_desc(),
                                         this->cpu_engine_, ctx);
          weight_data =
              reinterpret_cast<T2*>(weight_mkl.GetOpMem().get_data_handle());
        }
      }
      std::shared_ptr<stream> cpu_stream;
      MklDnnThreadPool eigen_tp(ctx);
      cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));

      // Temporary tensor for scaled bias when op is quantized version.
      Tensor temp_scaled_bias_tensor;
      if (std::is_same<T2, qint8>::value) {
        this->GetScaledBias(ctx, matmul_pd, bias_tensor,
                            &temp_scaled_bias_tensor, &bias_data);
      }

      // Execute fused matmul op.
      matmul_prim->Execute(src_data, weight_data, bias_data, dst_data,
                           cpu_stream);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  virtual void ExtendMklDnnMatMulFwdParams(OpKernelContext* ctx,
                                           MklDnnMatMulFwdParams& params) {
    // Create a string from data types of input, weight, bias, and output.
    params.dtypes.append(typeid(T1).name());
    params.dtypes.append(typeid(T2).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());

    if (fused_ops_.size() == 2) {
      string post_op = fused_ops_[1];

      if (post_op == "Relu") {
        params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
      } else if (post_op == "Relu6") {
        params.post_op_params.push_back({"relu6", {1.0, 6.0, 0.0}});
      } else if (post_op == "Elu") {
        params.post_op_params.push_back({"elu", {1.0, 1.0, 0.0}});
      } else if (post_op == "GeluApproximate") {
        params.post_op_params.push_back({"gelu_approximate", {1.0, 1.0, 0.0}});
      } else if (post_op == "GeluExact") {
        params.post_op_params.push_back({"gelu_exact", {1.0, 1.0, 0.0}});
      } else if (post_op == "Tanh") {
        params.post_op_params.push_back({"tanh", {1.0, 0.0, 0.0}});
      } else if (post_op == "Add") {
        params.post_op_params.push_back({"sum", {1.0}});
      } else if (post_op == "LeakyRelu") {
        params.post_op_params.push_back(
            {"leakyrelu", {1.0, leakyrelu_alpha, 0.0}});
      } else {
        OP_REQUIRES_OK(
            ctx, errors::InvalidArgument(
                     "Unsupported post-argument in MklFusedMatMul: ", post_op));
      }
    }
  }

  virtual void GetScaledBias(
      OpKernelContext* ctx,
      std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& matmul_pd,
      const Tensor& bias_tensor, Tensor* temp_scaled_bias_tensor,
      Tbias** bias_data) {}

 protected:
  std::vector<string> fused_ops_;
  bool transpose_a_;
  bool transpose_b_;

 private:
  bool fuse_add_ = false;
  float leakyrelu_alpha = 0.2;
  int kInputIndex_Add = 3;
  const int kOutputIndex_Dst = 0;
};

template <typename Device, typename T1, typename T2, typename Tbias,
          typename Toutput, bool native_format = false>
class MklQuantizedFusedMatMulOp
    : public MklFusedMatMulOp<Device, T1, T2, Tbias, Toutput, native_format> {
 public:
  explicit MklQuantizedFusedMatMulOp(OpKernelConstruction* context)
      : MklFusedMatMulOp<Device, T1, T2, Tbias, Toutput, native_format>(
            context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_bias_const", &(this->is_bias_const_)));
    OP_REQUIRES_OK(context, context->GetAttr("input_quant_mode", &mode_));
  }

  void Compute(OpKernelContext* ctx) override {
    MklFusedMatMulOp<Device, T1, T2, Tbias, Toutput, native_format>::Compute(
        ctx);
    // Compute additional outputs
    if (std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint32>::value) {
      Tensor* min_output = nullptr;
      Tensor* max_output = nullptr;
      MklDnnShape mkl_shape_min_output;
      MklDnnShape mkl_shape_max_output;
      mkl_shape_min_output.SetMklTensor(false);
      mkl_shape_max_output.SetMklTensor(false);
      AllocateOutputSetMklShape(ctx, 1, &min_output, {}, mkl_shape_min_output);
      AllocateOutputSetMklShape(ctx, 2, &max_output, {}, mkl_shape_max_output);
      if (std::is_same<Toutput, qint32>::value) {
        // Allowed fusions are (i) BiasAdd and (ii) BiasAdd + Relu. Other
        // activation being non-linear is ill-formed, since min-max are not
        // linear in intermediate output of MatMul.
        OP_REQUIRES(
            ctx,
            (this->fused_ops_.size() == 1) /*BiasAdd*/ ||
                (this->fused_ops_.size() == 2 && this->fused_ops_[1] == "Relu"),
            errors::InvalidArgument("Unsupported fusion."));
        const float min_input =
            ctx->input(kInputIndexMinInput).flat<float>()(0);
        const float max_input =
            ctx->input(kInputIndexMaxInput).flat<float>()(0);
        const float min_weight =
            ctx->input(kInputIndexMinWeight).flat<float>()(0);
        const float max_weight =
            ctx->input(kInputIndexMaxWeight).flat<float>()(0);
        float min_output_value;
        float max_output_value;
        MklQuantizationRangeForMultiplication<T1, T2, qint32>(
            min_input, max_input, min_weight, max_weight, &min_output_value,
            &max_output_value);
        min_output->flat<float>()(0) = min_output_value;
        max_output->flat<float>()(0) = max_output_value;
      } else {
        // When output type is qint8 or quint8, the kernel is registered for
        // Requantize fusion and desired output min-max are inputs to the
        // kernel.
        min_output->flat<float>()(0) =
            ctx->input(kInputIndexMinOutput).flat<float>()(0);
        max_output->flat<float>()(0) =
            ctx->input(kInputIndexMaxOutput).flat<float>()(0);
      }
    } else if (std::is_same<Toutput, float>::value) {
      // Kernel is registered for Dequantization fusion. Nothing to do.
    } else {
      OP_REQUIRES_OK(ctx, errors::InvalidArgument("Unsupported output type"));
    }
  }

  void ExtendMklDnnMatMulFwdParams(OpKernelContext* ctx,
                                   MklDnnMatMulFwdParams& params) override {
    // Create a string from data types of input, weight, bias, and output.
    params.dtypes.append(typeid(T1).name());
    params.dtypes.append(typeid(T2).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());

    if (std::is_same<Toutput, qint32>::value) {
      if (this->fused_ops_.size() == 1) {
        return;
      } else if (this->fused_ops_.size() == 2 &&
                 this->fused_ops_[1] == "Relu") {
        params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        return;
      } else {
        OP_REQUIRES_OK(ctx, errors::InvalidArgument(
                                "Activation other than Relu is not supported, "
                                "when quantized fused op has output data-type "
                                "as qint32. Fused op got activation function: ",
                                this->fused_ops_[1]));
      }
    } else if (std::is_same<Toutput, qint8>::value ||
               std::is_same<Toutput, quint8>::value ||
               std::is_same<Toutput, float>::value) {
      // When Toutput is float, the fusion semantic has its output dequantized,
      // and when Toutput is q{u}int8 the fusion semantic has its output
      // requantized.
      const float min_input = ctx->input(kInputIndexMinInput).flat<float>()(0);
      const float max_input = ctx->input(kInputIndexMaxInput).flat<float>()(0);
      const Tensor& min_weight_tensor = ctx->input(kInputIndexMinWeight);
      const Tensor& max_weight_tensor = ctx->input(kInputIndexMaxWeight);
      const float* min_weight = min_weight_tensor.flat<float>().data();
      const float* max_weight = max_weight_tensor.flat<float>().data();
      const size_t num_output_channels = min_weight_tensor.NumElements();

      const float max_int8_input =
          (std::is_same<T1, quint8>::value) ? 255.0f : 127.0f;
      const float max_int8_weight =
          (std::is_same<T2, quint8>::value) ? 255.0f : 127.0f;
      const float range_input =
          (mode_ == "MIN_FIRST")
              ? max_input - min_input
              : std::max(std::abs(min_input), std::abs(max_input));

      std::vector<float> scale_output(num_output_channels);
      for (size_t i = 0; i < num_output_channels; ++i) {
        float range_weight =
            std::max(std::abs(min_weight[i]), std::abs(max_weight[i]));
        scale_output[i] =
            (range_input * range_weight) / (max_int8_input * max_int8_weight);
      }

      float scale_post_op = 1.0;
      if (!std::is_same<Toutput, float>::value) {
        const float min_output =
            ctx->input(kInputIndexMinOutput).flat<float>()(0);
        const float max_output =
            ctx->input(kInputIndexMaxOutput).flat<float>()(0);
        const float range_output =
            std::max(std::abs(min_output), std::abs(max_output));

        if (std::is_same<Toutput, qint8>::value) {
          if (this->fused_ops_.size() == 1) {
            for (size_t i = 0; i < scale_output.size(); ++i) {
              scale_output[i] = scale_output[i] * (127.0f / range_output);
            }
            FactoryKeyCreator partial_key;
            partial_key.AddAsKey<float>(range_input);
            partial_key.AddAsKey<const float*>(min_weight);
            partial_key.AddAsKey<const float*>(max_weight);
            params.post_op_params.push_back(
                {"output_scale", scale_output, partial_key.GetKey()});
            return;
          }
          scale_post_op = 127.0f / range_output;
        } else if (std::is_same<Toutput, quint8>::value) {
          // This is only supported when activation is Relu or Relu6.
          OP_REQUIRES(
              ctx,
              (this->fused_ops_.size() == 2 &&
               (this->fused_ops_[1] == "Relu" ||
                this->fused_ops_[1] == "Relu6")),
              errors::InvalidArgument(
                  "Unsupported QuantizedFusedMatMul with output type quint8"));
          scale_post_op = 255.0f / range_output;
        }
      }

      // Dequantize
      FactoryKeyCreator partial_key;
      partial_key.AddAsKey<float>(range_input);
      partial_key.AddAsKey<const float*>(min_weight);
      partial_key.AddAsKey<const float*>(max_weight);
      params.post_op_params.push_back(
          {"output_scale", scale_output, partial_key.GetKey()});

      // Apply activation, if any, after Dequantize. Note scale_post_op along
      // with output type would take care of both Dequantize and Requantize
      // fusion.
      if (this->fused_ops_.size() == 2) {
        string activation = this->fused_ops_[1];
        if (activation == "Relu") {
          params.post_op_params.push_back({"relu", {scale_post_op, 0.0, 0.0}});
        } else if (activation == "Relu6") {
          params.post_op_params.push_back({"relu6", {scale_post_op, 6.0, 0.0}});
        } else if (activation == "Elu") {
          params.post_op_params.push_back({"elu", {scale_post_op, 1.0, 0.0}});
        } else if (activation == "GeluApproximate") {
          params.post_op_params.push_back(
              {"gelu_approximate", {scale_post_op, 1.0, 0.0}});
        } else if (activation == "GeluExact") {
          params.post_op_params.push_back(
              {"gelu_exact", {scale_post_op, 1.0, 0.0}});
        } else if (activation == "Tanh") {
          params.post_op_params.push_back({"tanh", {scale_post_op, 0.0, 0.0}});
        } /* temporary hack*/ else if (activation == "Add") {
          params.post_op_params.push_back({"sum", {1.0}});
        } else {
          OP_REQUIRES_OK(ctx,
                         errors::InvalidArgument(
                             "Unsupported Activation in QuantizedFusedMatMul: ",
                             activation));
        }
      }
    } else {
      OP_REQUIRES_OK(ctx,
                     errors::InvalidArgument(
                         "Unsupported output type in QuantizedFusedMatMul."));
    }
  }

  void GetScaledBias(
      OpKernelContext* ctx,
      std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>& matmul_pd,
      const Tensor& bias_tensor, Tensor* temp_scaled_bias_tensor,
      Tbias** bias_data) override {
    const float min_input = ctx->input(kInputIndexMinInput).flat<float>()(0);
    const float max_input = ctx->input(kInputIndexMaxInput).flat<float>()(0);
    const Tensor& min_weight_tensor = ctx->input(kInputIndexMinWeight);
    const Tensor& max_weight_tensor = ctx->input(kInputIndexMaxWeight);
    const float* min_weight = min_weight_tensor.flat<float>().data();
    const float* max_weight = max_weight_tensor.flat<float>().data();

    // We can use cached bias which has been scaled only when (i) bias is
    // constant (ii) weight is constant (iii) min_input is same as saved one
    // (iv) max_input is same as saved one (v) BiasCache is not empty.
    if (!this->IsBiasCacheEmpty() &&
        this->IsCachedBiasValid(min_input, max_input)) {
      this->GetCachedBias(ctx, bias_data);
    } else {
      void* input_bias_buf = static_cast<void*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      auto scaled_bias_md = matmul_pd->bias_desc();
      TensorShape scaled_bias_shape;
      scaled_bias_shape.AddDim((scaled_bias_md.get_size() / sizeof(Tbias)));
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<Tbias>::v(), scaled_bias_shape,
                                  temp_scaled_bias_tensor));
      void* scaled_bias_buf =
          static_cast<void*>(temp_scaled_bias_tensor->flat<Tbias>().data());

      const float max_int8_input =
          (std::is_same<T1, quint8>::value) ? 255.0f : 127.0f;
      const float max_int8_weight =
          (std::is_same<T2, quint8>::value) ? 255.0f : 127.0f;
      const float range_input =
          (mode_ == "MIN_FIRST")
              ? max_input - min_input
              : std::max(std::abs(min_input), std::abs(max_input));
      const size_t num_weight_scales = min_weight_tensor.NumElements();
      std::vector<float> bias_scales(num_weight_scales, 1.0);
      for (size_t i = 0; i < num_weight_scales; ++i) {
        float range_weight =
            std::max(std::abs(min_weight[i]), std::abs(max_weight[i]));
        float scale_factor =
            (max_int8_input * max_int8_weight) / (range_input * range_weight);
        bias_scales[i] = scale_factor;
      }
      if (mode_ == "MIN_FIRST") {
        Tbias* input_bias = (Tbias*)input_bias_buf;
        Tbias* adjusted_bias = (Tbias*)scaled_bias_buf;
        float q_min_input = max_int8_input * min_input / range_input;
        const Tensor& weight_tensor = ctx->input(1);
        int k = weight_tensor.dim_size(0);
        int n = weight_tensor.dim_size(1);
        T2* wt_buf = const_cast<T2*>(weight_tensor.flat<T2>().data());
        // Scales needs to expanded to number of output channels by the values
        // of bias_scales.
        std::vector<float> scales(n);
        if (num_weight_scales == 1)  // weights quanitzed per_tensor
          std::fill(scales.begin(), scales.end(), bias_scales[0]);
        else
          scales = bias_scales;  // Expensive copy
#pragma omp parallel for schedule(static)
        for (int j = 0; j < n; ++j) {
          int sum = 0;
          for (int i = 0; i < k; ++i) {
            sum += wt_buf[i * n + j];
          }
          adjusted_bias[j] = ((input_bias[j] * scales[j]) +
                              static_cast<float>(sum * q_min_input));
        }
      } else {
        mkldnn::primitive_attr bias_attr;
        (num_weight_scales == 1) ? bias_attr.set_output_scales(0, bias_scales)
                                 : bias_attr.set_output_scales(1, bias_scales);
        memory::dims input_bias_dims =
            memory::dims({bias_tensor.shape().dim_size(0)});
        auto input_bias_md = mkldnn::memory::desc(
            input_bias_dims, MklDnnType<Tbias>(), memory::format_tag::x);
        auto input_bias_mem =
            mkldnn::memory(input_bias_md, this->cpu_engine_, input_bias_buf);
        auto scaled_bias_mem =
            mkldnn::memory(scaled_bias_md, this->cpu_engine_, scaled_bias_buf);
        auto reorder_prim =
            mkldnn::reorder(input_bias_mem, scaled_bias_mem, bias_attr);
        std::unordered_map<int, memory> reorder_net_args = {
            {MKLDNN_ARG_FROM, input_bias_mem},
            {MKLDNN_ARG_TO, scaled_bias_mem}};
        reorder_prim.execute(mkldnn::stream(this->cpu_engine_),
                             reorder_net_args);
      }

      *bias_data = temp_scaled_bias_tensor->flat<Tbias>().data();

      // Cache the scaled bias
      if (this->is_bias_const_ && this->is_weight_const_) {
        this->CacheBias(ctx, *temp_scaled_bias_tensor);
        this->saved_min_input_ = min_input;
        this->saved_max_input_ = max_input;
      }
    }
  }

  bool IsCachedBiasValid(float current_min_input, float current_max_input) {
    if (this->is_bias_const_ && this->is_weight_const_ &&
        std::abs(current_min_input - saved_min_input_) < 1e-5 &&
        std::abs(current_max_input - saved_max_input_) < 1e-5)
      return true;
    return false;
  }

 private:
  const int kInputIndexMinInput = 3;
  const int kInputIndexMaxInput = 4;
  const int kInputIndexMinWeight = 5;
  const int kInputIndexMaxWeight = 6;
  const int kInputIndexMinOutput = 7;
  const int kInputIndexMaxOutput = 8;

  string mode_;
  float saved_min_input_ = -std::numeric_limits<float>::infinity();
  float saved_max_input_ = std::numeric_limits<float>::infinity();
};

// Register mkl kernels for supported operations and types.
#define REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES(type) \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedMatMul")                                  \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklFusedMatMulOp<CPUDevice, type, type, type, type>);    \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklNativeFusedMatMul")                            \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),      \
      MklFusedMatMulOp<CPUDevice, type, type, type, type, true>);
TF_CALL_float(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
#undef REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES

REGISTER_KERNEL_BUILDER(Name("_QuantizedFusedMatMul")
                            .Device(DEVICE_CPU)
                            .TypeConstraint("T1", {DT_QUINT8, DT_QINT8})
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint("Targs", {DT_FLOAT, DT_QUINT8,
                                                      DT_QINT8, DT_QINT32})
                            .TypeConstraint<qint32>("Toutput"),
                        NoOp);

REGISTER_KERNEL_BUILDER(Name("_QuantizedFusedMatMulAndDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint("T1", {DT_QUINT8, DT_QINT8})
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint("Targs", {DT_FLOAT, DT_QUINT8,
                                                      DT_QINT8, DT_QINT32})
                            .TypeConstraint<float>("Toutput"),
                        NoOp);

REGISTER_KERNEL_BUILDER(Name("_QuantizedFusedMatMulAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint("T1", {DT_QUINT8, DT_QINT8})
                            .TypeConstraint<qint8>("T2")
                            .TypeConstraint("Targs", {DT_FLOAT, DT_QUINT8,
                                                      DT_QINT8, DT_QINT32})
                            .TypeConstraint("Toutput", {DT_QUINT8, DT_QINT8}),
                        NoOp);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMul")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<qint32>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, quint8, qint8, float, qint32>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMul")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<qint32>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, qint8, qint8, float, qint32>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMulAndDequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<float>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, quint8, qint8, float, float>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMulAndDequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<float>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, qint8, qint8, float, float>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMulAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<qint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, quint8, qint8, float, qint8>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMulAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<qint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, qint8, qint8, float, qint8>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMulAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<quint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, quint8, qint8, float, quint8>);

REGISTER_KERNEL_BUILDER(
    Name("_MklQuantizedFusedMatMulAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<qint8>("T1")
        .TypeConstraint<qint8>("T2")
        .TypeConstraint<float>("Targs")
        .TypeConstraint<quint8>("Toutput")
        .Label(mkl_op_registry::kMklQuantizedOpLabel),
    MklQuantizedFusedMatMulOp<CPUDevice, qint8, qint8, float, quint8>);

}  // namespace tensorflow

#endif  // INTEL_MKL
