#include <algorithm>
#include <vector>
#include <cmath>
#include "caffe/layers/softmaxl2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void Softmaxl2LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	softmaxVals_.ReshapeLike(*bottom[0]);
	label_sub_softmaxVals_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void Softmaxl2LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

 
  //CHECK_EQ(bottom[0]->channels(), 1);
  //CHECK_EQ(bottom[1]->channels(), 1);
}

template <typename Dtype>
void Softmaxl2LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	int batchSize = bottom[0]->count(0, 1);
	int probDim = bottom[0]->count(1);

	Blob<Dtype> tmpA, tmpB;
	Blob<Dtype> maxVals;

	tmpA.Reshape(bottom[0]->shape());
	tmpB.Reshape(bottom[0]->shape());

	maxVals.ReshapeLike(*bottom[1]);

	for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
	{
		Dtype maxVal = bottom[0]->cpu_data()[0];
		const Dtype* prob = bottom[0]->cpu_data() + sampleIdx * probDim;
		for (int probIdx = 1; probIdx < probDim; probIdx++)
		{
			if (maxVal < prob[probIdx])
				maxVal = prob[probIdx];
		}
		maxVals.mutable_cpu_data()[sampleIdx] = maxVal;
		caffe_set<Dtype>(probDim, maxVal, tmpA.mutable_cpu_data() + sampleIdx * probDim);
	}
	caffe_sub<Dtype>(bottom[0]->count(), bottom[0]->cpu_data(), tmpA.cpu_data(), tmpB.mutable_cpu_data());
	caffe_exp<Dtype>(bottom[0]->count(), tmpB.cpu_data(), softmaxVals_.mutable_cpu_data()); //exp(x)

	for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
	{
		const Dtype* prob = softmaxVals_.cpu_data() + sampleIdx * probDim;
		Dtype sum = caffe_cpu_asum<Dtype>(probDim, prob);
		if (sum < FLT_MIN)
			sum = FLT_MIN;
		caffe_set<Dtype>(probDim, sum, tmpA.mutable_cpu_data() + sampleIdx * probDim);
	}
	caffe_div<Dtype>(bottom[0]->count(), softmaxVals_.cpu_data(), tmpA.cpu_data(), tmpB.mutable_cpu_data()); //normalizing sum to 1
	caffe_copy<Dtype>(bottom[0]->count(), tmpB.cpu_data(), softmaxVals_.mutable_cpu_data());

	Dtype lossVal = 0;
	for (int sampleIdx = 0; sampleIdx < batchSize; sampleIdx++)
	{
		const int label = (int)(bottom[1]->cpu_data()[sampleIdx]);
		for (int probIdx = 0; probIdx < probDim; probIdx++)
		{
			Dtype val = 0;
			if (probIdx == label)
				val = 1 - softmaxVals_.cpu_data()[sampleIdx * probDim + probIdx];
			else
				val = 0 - softmaxVals_.cpu_data()[sampleIdx * probDim + probIdx];
			label_sub_softmaxVals_.mutable_cpu_data()[sampleIdx * probDim + probIdx] = val;
			lossVal += val * val; //pow(softmax,2)
		}
	}
	top[0]->mutable_cpu_data()[0] = lossVal / batchSize;
	return;
}

template <typename Dtype>
void Softmaxl2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        
  Blob<Dtype> tmpA,tmpB, tmpC;
  tmpA.Reshape( bottom[0]->shape() );
  tmpB.Reshape( bottom[0]->shape() );
  tmpC.Reshape( bottom[0]->shape() );
  caffe_mul<Dtype>(bottom[0]->count(), label_sub_softmaxVals_.cpu_data(), softmaxVals_.cpu_data(), tmpA.mutable_cpu_data()); //f(x) * (y - f(x))


  caffe_set<Dtype>(bottom[0]->count(), Dtype(1), tmpB.mutable_cpu_data());
  caffe_sub<Dtype>(bottom[0]->count(), tmpB.cpu_data(), softmaxVals_.cpu_data(), tmpC.mutable_cpu_data()); //1 - f(x)
  
  caffe_mul<Dtype>(bottom[0]->count(), tmpA.cpu_data(), tmpC.cpu_data(), tmpB.mutable_cpu_data()); //f(x) * (y - f(x)) * (1 - f(x))
  
  caffe_set<Dtype>(bottom[0]->count(), Dtype(-2), tmpA.mutable_cpu_data());
  caffe_mul<Dtype>(bottom[0]->count(), tmpA.cpu_data(), tmpB.cpu_data(), bottom[0]->mutable_cpu_diff());

  Dtype lossW = top[0]->cpu_diff()[0] / bottom[0]->count(0, 1); //batch size
  caffe_scal<Dtype>(bottom[0]->count(), lossW, bottom[0]->mutable_cpu_diff());
  return;
}

#ifdef CPU_ONLY
STUB_GPU(Softmaxl2LossLayer);
#endif

INSTANTIATE_CLASS(Softmaxl2LossLayer);
REGISTER_LAYER_CLASS(Softmaxl2Loss);

}  // namespace caffe
