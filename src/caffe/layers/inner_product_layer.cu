#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

template <typename Dtype>
__global__ void pos_kernel(const int n, const Dtype* a, Dtype* b) {
  CUDA_KERNEL_LOOP(index, n) {
    if (a[index] > 0)
      b[index] = a[index];
  }
}

template <typename Dtype>
__global__ void div_r_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    if (b[index] != 0)
      y[index] = a[index] / b[index];
    else
      y[index] = 0;
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_eb_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // get the new weight W+
    const Dtype* W_data = this->blobs_[0]->gpu_data();
    Blob<Dtype> W_plus(this->blobs_[0]->shape());
    Dtype* W_plus_data = W_plus.mutable_gpu_data();
    caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
    pos_kernel<Dtype><<<CAFFE_GET_BLOCKS(W_plus.count()), CAFFE_CUDA_NUM_THREADS>>>(
          W_plus.count(), W_data, W_plus_data);
  
    // compute the normalization factor by forwardpassing using W+
    Blob<Dtype> NN(top[0]->shape());  
    Dtype* NN_data = NN.mutable_gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    
    if (M_ == 1) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                           W_plus_data, bottom_data, (Dtype)0., NN_data);
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans,
                           transpose_ ? CblasNoTrans : CblasTrans,
                           M_, N_, K_, (Dtype)1.,
                           bottom_data, W_plus_data, (Dtype)0., NN_data);
    }
  
    // do normalization
    const Dtype* top_diff = top[0]->gpu_diff();
    div_r_kernel<Dtype><<<CAFFE_GET_BLOCKS(NN.count()), CAFFE_CUDA_NUM_THREADS>>>(
          NN.count(), top_diff, NN_data, NN_data);
  
    // do backwardpass  
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., NN_data, W_plus_data,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., NN_data, W_plus_data,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  
    // multiply the bottom data
    caffe_gpu_mul<Dtype>(bottom[0]->count(), bottom[0]->gpu_diff(), bottom_data, bottom[0]->mutable_gpu_diff());
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);
INSTANTIATE_LAYER_EB_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
