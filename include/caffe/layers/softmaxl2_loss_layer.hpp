#ifndef CAFFE_SOFTMAXL2_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAXL2_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class Softmaxl2LossLayer : public LossLayer<Dtype> {
 public:
  explicit Softmaxl2LossLayer(const LayerParameter& param)
	  : LossLayer<Dtype>(param) {   }
  virtual inline const char* type() const { return "Softmaxl2Loss"; }

protected:
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 protected:
	 Blob<Dtype> softmaxVals_, label_sub_softmaxVals_;
};


}  // namespace caffe

#endif  // CAFFE_CORRELATION_LOSS_LAYER_HPP_
