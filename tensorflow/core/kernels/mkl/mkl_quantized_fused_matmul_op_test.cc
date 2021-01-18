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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

static const uint8 dummy_tensor[] = {0, 0, 0, 0, 0, 0, 0, 0};
static const TensorShape dummy_shape({8});

template <typename T>
class CommonTestUtilities : public OpsTestBase {
 public:
  void MklToTF(const Tensor& tensor, const Tensor& mkl_meta_tensor,
               Tensor* output) {
    // Create an MKL to TF conversion node and execute it
    TF_ASSERT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(DataTypeToEnum<T>::v()))
                     .Input(FakeInput(DT_UINT8))  // MKL second tensor
                     .Attr("T", DataTypeToEnum<T>::v())
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    AddInputFromArray<T>(tensor.shape(), tensor.flat<T>());
    AddInputFromArray<uint8>(mkl_meta_tensor.shape(),
                             mkl_meta_tensor.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    *output = *GetOutput(0);
  }

  void ConvertAndCompare(const Tensor& result, const Tensor& mkl_meta_tensor,
                         const Tensor& expected, float rtol = 0.01) {
    Tensor converted_output;
    MklToTF(result, mkl_meta_tensor, &converted_output);
    test::ExpectClose(expected, converted_output, /*atol=*/1e-5, rtol);
  }

  void ConvertAndCompareIntegral(const Tensor& result,
                                 const Tensor& mkl_meta_tensor,
                                 const Tensor& expected) {
    Tensor converted_output;
    MklToTF(result, mkl_meta_tensor, &converted_output);
    test::ExpectTensorEqual<T>(expected, converted_output);
  }

  void TestBody() {}
};

class QuantizedFusedMatMulTest : public OpsTestBase {
 public:
  void CreateNodeDef(NodeDefBuilder* node_def_builder, DataType input_dtype,
                     const std::vector<string>& fused_ops,
                     DataType bias_dtype = DT_FLOAT) {
    node_def_builder->Input(FakeInput(input_dtype))
        .Input(FakeInput(DT_QINT8))
        .Input(FakeInput(1, bias_dtype))
        .Input(FakeInput(DT_FLOAT))
        .Input(FakeInput(DT_FLOAT))
        .Input(FakeInput(DT_FLOAT))
        .Input(FakeInput(DT_FLOAT))
        .Input(FakeInput(DT_UINT8))  // MKL second tensor
        .Input(FakeInput(DT_UINT8))  // MKL second tensor
        .Input(FakeInput(DT_UINT8))  // MKL second tensor
        .Input(FakeInput(DT_UINT8))  // MKL second tensor
        .Input(FakeInput(DT_UINT8))  // MKL second tensor
        .Input(FakeInput(DT_UINT8))  // MKL second tensor
        .Input(FakeInput(DT_UINT8))  // MKL second tensor
        .Attr("Targs", DT_FLOAT)
        .Attr("Toutput", DataTypeToEnum<qint32>::v())
        .Attr("T", DT_QINT32)
        .Attr("transpose_a", false)
        .Attr("transpose_b", false)
        .Attr("num_args", 1)
        .Attr("is_filter_const", true)
        .Attr("is_bias_const", true)
        .Attr("fused_ops", fused_ops)
        .Attr("_kernel", "QuantizedMklOp");

    TF_ASSERT_OK(node_def_builder->Finalize(node_def()));
  }
};

// The following two tests are for _MklQuantizedFusedMatMul op with 32-bit
// integer output. The op is valid only for fusions (i) BiasAdd and (ii) BiasAdd
// + Relu. Other activations such as Elu, Tanh, Gelu etc. are ill-formed because
// min-max range of fusion output is no longer linear the in 32-bit intermediate
// result of MatMul.
TEST_F(QuantizedFusedMatMulTest, UnsignedInputBiasAddRelu) {
  NodeDefBuilder node_def_builder = NodeDefBuilder(
      "mkl_quantized_fused_matmul_bias", "_MklQuantizedFusedMatMul");
  // BiasAdd only
  CreateNodeDef(&node_def_builder, DT_QUINT8, {"BiasAdd"});
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  1 |  2 |  3 |
  // |  4 |  5 |  6 |
  AddInputFromArray<quint8>(TensorShape({2, 3}), {1, 2, 3, 4, 5, 6});
  // B matrix is:
  // |  7 |  -8 |  -9 | 10 |
  // | 11 | -12 | -13 | 14 |
  // | 15 | -16 | -17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, -8, -9, 10, 11, -12, -13, 14, 15, -16, -17, 18});
  // bias vector {1, 2, 3, 4}
  AddInputFromArray<float>(TensorShape({4}), {1, 2, -3, -4});
  // Min and Max values for input and weights are set to min and max values of
  // the respective 8-bit integral data types so that q{u}int8 and float
  // representations are identical in values, i.e., scaling factor for
  // quantization is 1.0.
  AddInputFromArray<float>(TensorShape({1}), {0.0f});
  AddInputFromArray<float>(TensorShape({1}), {255.0f});
  AddInputFromArray<float>(TensorShape({1}), {-127.0f});
  AddInputFromArray<float>(TensorShape({1}), {127.0f});
  int num_input_tensors = this->tensors_.size();
  for (int i = 0; i < num_input_tensors; ++i)
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

  using KernelType = MklDnnMatMulOpBase<qint8, float, qint32>;
  // Before first time kernel execution, weight and bias cache should be empty
  EXPECT_TRUE(static_cast<KernelType*>(this->kernel_.get())
                  ->IsWeightCacheEmpty(this->context_.get()));
  EXPECT_TRUE(
      static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty());
  EXPECT_FALSE(
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (1 * 7)  + (2 * 11)  + (3 * 15) = 74
  // (1 * -8) + (2 * -12) + (3 * -16)  = -80
  // (1 * -9) + (2 * -13) + (3 * -17)  = -86
  // (1 * 10) + (2 * 14)  + (3 * 18) = 92
  // (4 * 7)  + (5 * 11)  + (6 * 15) = 173
  // (4 * -8) + (5 * -12) + (6 * -16)  = -188
  // (4 * -9) + (5 * -13) + (6 * -17)  = -203
  // (4 * 10) + (5 * 14)  + (6 * 18) = 218
  // Final result after Bias addition:
  // 74  + 1 = 75 , -80  + 2 = -78 , -86 - 3 = -89 , 92 - 4 = 88,
  // 173 + 1 = 174, -188 + 2 = -186, -203 - 3 = -206, 218 - 4 = 214
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 4}));
  test::FillValues<qint32>(&expected, {75, -78, -89, 88, 174, -186, -206, 214});
  const Tensor& output = *GetOutput(0);
  const Tensor& mkl_shape_tensor = *GetOutput(3);
  CommonTestUtilities<qint32> util;
  util.ConvertAndCompareIntegral(output, mkl_shape_tensor, expected);

  // After first time kernel execution, bias cache should not be empty. The
  // weight, however is not guaranteed to be cached since weight caching depends
  // on inner_product primitive descriptor.
  EXPECT_TRUE(
      (!static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty()) &&
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  // Execute kernel with cached bias.
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_caching = *GetOutput(0);
  const Tensor& mkl_shape_tensor_caching = *GetOutput(3);
  CommonTestUtilities<qint32> util_caching;
  util_caching.ConvertAndCompareIntegral(output_caching,
                                         mkl_shape_tensor_caching, expected);

  // BiasAdd + Relu
  auto attr_map = this->node_def_.mutable_attr();
  attr_map->erase(attr_map->find("fused_ops"));
  AddNodeAttr("fused_ops", {"BiasAdd", "Relu"}, &(this->node_def_));
  TF_ASSERT_OK(InitOp());  // Overwrites older kernel with new one.
  // Before first time kernel execution, weight and bias cache should be empty.
  EXPECT_TRUE(static_cast<KernelType*>(this->kernel_.get())
                  ->IsWeightCacheEmpty(this->context_.get()));
  EXPECT_TRUE(
      static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty());
  EXPECT_FALSE(
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& relu_output = *GetOutput(0);
  const Tensor& relu_mkl_shape_tensor = *GetOutput(3);
  // Exepected values after Relu
  test::FillValues<qint32>(&expected, {75, 0, 0, 88, 174, 0, 0, 214});
  CommonTestUtilities<qint32> relu_util;
  relu_util.ConvertAndCompareIntegral(relu_output, relu_mkl_shape_tensor,
                                      expected);

  // After first time kernel execution, bias cache should not be empty. The
  // weight, however is not guaranteed to be cached since weight caching depends
  // on inner_product primitive descriptor.
  EXPECT_TRUE(
      (!static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty()) &&
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  // Execute kernel with cached bias.
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& relu_output_caching = *GetOutput(0);
  const Tensor& relu_mkl_shape_tensor_caching = *GetOutput(3);
  CommonTestUtilities<qint32> relu_util_caching;
  relu_util_caching.ConvertAndCompareIntegral(
      relu_output_caching, relu_mkl_shape_tensor_caching, expected);
}

TEST_F(QuantizedFusedMatMulTest, SignedInputBiasAddRelu) {
  NodeDefBuilder node_def_builder = NodeDefBuilder(
      "mkl_quantized_fused_matmul_bias", "_MklQuantizedFusedMatMul");
  // BiasAdd only
  CreateNodeDef(&node_def_builder, DT_QINT8, {"BiasAdd"});
  TF_ASSERT_OK(InitOp());
  // A matrix is:
  // |  -1 |  2 |  -3 |
  // |  -4 |  5 |  -6 |
  AddInputFromArray<qint8>(TensorShape({2, 3}), {-1, 2, -3, -4, 5, -6});
  // B matrix is:
  // |  7 |  -8 |  -9 | 10 |
  // | 11 | -12 | -13 | 14 |
  // | 15 | -16 | -17 | 18 |
  AddInputFromArray<qint8>(TensorShape({3, 4}),
                           {7, -8, -9, 10, 11, -12, -13, 14, 15, -16, -17, 18});
  // bias vector {1, 2, 3, 4}
  AddInputFromArray<float>(TensorShape({4}), {1, 2, -3, -4});
  // Min and Max values for input and weights are set to min and max values of
  // the respective 8-bit integral data types so that q{u}int8 and float
  // representations are identical in values, i.e., scaling factor for
  // quantization is 1.0.
  AddInputFromArray<float>(TensorShape({1}), {-127.0f});
  AddInputFromArray<float>(TensorShape({1}), {127.0f});
  AddInputFromArray<float>(TensorShape({1}), {-127.0f});
  AddInputFromArray<float>(TensorShape({1}), {127.0f});
  int num_input_tensors = this->tensors_.size();
  for (int i = 0; i < num_input_tensors; ++i)
    AddInputFromArray<uint8>(dummy_shape, dummy_tensor);

  using KernelType = MklDnnMatMulOpBase<qint8, float, qint32>;
  // Before first time kernel execution, weight and bias cache should be empty
  EXPECT_TRUE(static_cast<KernelType*>(this->kernel_.get())
                  ->IsWeightCacheEmpty(this->context_.get()));
  EXPECT_TRUE(
      static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty());
  EXPECT_FALSE(
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  TF_ASSERT_OK(RunOpKernel());
  // Here are the results we expect, from hand calculations:
  // (-1 * 7) + (2 * 11)  + (-3 * 15) = -30
  // (-1 * -8) + (2 * -12) + (-3 * -16)  = 32
  // (-1 * -9) + (2 * -13) + (-3 * -17)  = 34
  // (-1 * 10) + (2 * 14)  + (-3 * 18) = -36
  // (-4 * 7)  + (5 * 11)  + (-6 * 15) = -63
  // (-4 * -8) + (5 * -12) + (-6 * -16)  = 68
  // (-4 * -9) + (5 * -13) + (-6 * -17)  = 73
  // (-4 * 10) + (5 * 14)  + (-6 * 18) = -78
  // Final result after Bias addition:
  // -30  + 1 = -29 , 32  + 2 = 34 , 34 - 3 = 31 , -36 - 4 = -40,
  // -63 + 1 = -62, 68 + 2 = 70, 73 - 3 = 70, -78 - 4 = -82
  Tensor expected(allocator(), DT_QINT32, TensorShape({2, 4}));
  test::FillValues<qint32>(&expected, {-29, 34, 31, -40, -62, 70, 70, -82});
  const Tensor& output = *GetOutput(0);
  const Tensor& mkl_shape_tensor = *GetOutput(3);
  CommonTestUtilities<qint32> util;
  util.ConvertAndCompareIntegral(output, mkl_shape_tensor, expected);

  // After first time kernel execution, bias cache should not be empty. The
  // weight, however is not guaranteed to be cached since weight caching depends
  // on inner_product primitive descriptor.
  EXPECT_TRUE(
      (!static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty()) &&
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  // Execute kernel with cached bias.
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& output_caching = *GetOutput(0);
  const Tensor& mkl_shape_tensor_caching = *GetOutput(3);
  CommonTestUtilities<qint32> util_caching;
  util_caching.ConvertAndCompareIntegral(output_caching,
                                         mkl_shape_tensor_caching, expected);

  // BiasAdd + Relu
  auto attr_map = this->node_def_.mutable_attr();
  attr_map->erase(attr_map->find("fused_ops"));
  AddNodeAttr("fused_ops", {"BiasAdd", "Relu"}, &(this->node_def_));
  TF_ASSERT_OK(InitOp());  // Overwrites older kernel with new one.
  // Before first time kernel execution, weight and bias cache should be empty.
  EXPECT_TRUE(static_cast<KernelType*>(this->kernel_.get())
                  ->IsWeightCacheEmpty(this->context_.get()));
  EXPECT_TRUE(
      static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty());
  EXPECT_FALSE(
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& relu_output = *GetOutput(0);
  const Tensor& relu_mkl_shape_tensor = *GetOutput(3);
  // Exepected values after BiasAdd and Relu
  test::FillValues<qint32>(&expected, {0, 34, 31, 0, 0, 70, 70, 0});
  CommonTestUtilities<qint32> relu_util;
  relu_util.ConvertAndCompareIntegral(relu_output, relu_mkl_shape_tensor,
                                      expected);

  // After first time kernel execution, bias cache should not be empty. The
  // weight, however is not guaranteed to be cached since weight caching depends
  // on inner_product primitive descriptor.
  EXPECT_TRUE(
      (!static_cast<KernelType*>(this->kernel_.get())->IsBiasCacheEmpty()) &&
      static_cast<KernelType*>(this->kernel_.get())->IsCachedBiasValid());
  // Execute kernel with cached bias.
  TF_ASSERT_OK(RunOpKernel());
  const Tensor& relu_output_caching = *GetOutput(0);
  const Tensor& relu_mkl_shape_tensor_caching = *GetOutput(3);
  CommonTestUtilities<qint32> relu_util_caching;
  relu_util_caching.ConvertAndCompareIntegral(
      relu_output_caching, relu_mkl_shape_tensor_caching, expected);
}
}  // namespace tensorflow

#endif  // INTEL_MKL
