#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<int, float> Prediction;

class Classifier {
 public:
	 Classifier(const string& model_file,
		 const string& trained_file,
		 const string& mean_rgb);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);
  void SetMean(float meanR, float meanG, float meanB);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
#if 0
  std::vector<string> labels_;
#endif
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
					   const string& mean_rgb) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
#if 1
  {
	  string tmpRGB = mean_rgb;
	  int pos = tmpRGB.find(',');
	  float meanR = atof(tmpRGB.substr(0, pos).c_str());
	  tmpRGB = tmpRGB.substr(pos + 1);
	  pos = tmpRGB.find(',');
	  float meanG = atof(tmpRGB.substr(0, pos).c_str());
	  tmpRGB = tmpRGB.substr(pos + 1);
	  float meanB = atof(tmpRGB.c_str());
	  SetMean(meanR, meanG, meanB);
  }
#else
  SetMean(mean_file);
#endif
  /* Load labels. */
#if 0
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
#endif
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

//  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(idx, output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}


void Classifier::SetMean(float meanR, float meanG, float meanB) {

	mean_ = cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(meanR, meanG, meanB));
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

#if 0
  cv::Mat sample_normalized = sample_float.clone();
#else
  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);
#endif

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  string model_file = "d:/dev/caffe-win/build/x64/Release/cifar/bvlc_alexnet/deploy.prototxt";
  string trained_file = "d:/dev/caffe-win/build/x64/Release/cifar/bvlc_alexnet/models/cifar_alexnet_iter_32400.caffemodel";
  string mean_rgb = "113.821,122.954,125.293"; //same order as train_val.prototxt
  string root_path = "g:/dataset/cifar/test/";
  string listfile = "g:/dataset/cifar/test.txt"; //same as list used in convert_imageset.exe
  int width_before_crop = 128;
  int height_before_crop = 128;
  int width_after_crop = 96;
  int height_after_crop = 96;
  Classifier classifier(model_file, trained_file, mean_rgb);


  std::vector<std::pair<std::string, int> > file_list;
  {
	  std::ifstream infile(listfile);
	  std::string line;
	  size_t pos;
	  int label;
	  while (std::getline(infile, line)) {
		  pos = line.find_last_of(' ');
		  label = atoi(line.substr(pos + 1).c_str());
		  file_list.push_back(std::make_pair(line.substr(0, pos), label));
	  }
  }

  std::vector< std::pair< std::string, bool > > test_list;
  for (int sample_idx = 0; sample_idx < file_list.size(); sample_idx++)
  {
	  string path = root_path + "//" + file_list[sample_idx].first;
	  cv::Mat img = cv::imread(path, 1);
	  if (width_before_crop > 0 || height_before_crop > 0)
	  {
		  cv::Mat resized;
		  cv::resize(img, resized, cv::Size(width_before_crop, height_before_crop));

		  int dx = (width_before_crop - width_after_crop) / 2;
		  int dy = (height_before_crop - height_after_crop) / 2;
		  cv::Rect roi(dx, dy, width_after_crop, height_after_crop);
		  img = resized(roi).clone();
	  }

	  CHECK(!img.empty()) << "Unable to decode image " << path;
	  std::vector<Prediction> predictions = classifier.Classify(img, 1);

	  /* Print the top N predictions. */
	  bool hit_flag = false;
	  for (size_t i = 0; i < predictions.size() && hit_flag == false; ++i) {
		  Prediction p = predictions[i];
		  hit_flag = p.first == file_list[sample_idx].second;
	  }
	  test_list.push_back(std::make_pair(file_list[sample_idx].first, hit_flag) );
  }

  std::map<int, int> class2num, class2hit;
  for (int k = 0; k < test_list.size(); k++)
  {
	  string path = test_list[k].first;
	  bool hit_flag = test_list[k].second;
	  int label = file_list[k].second;
	  CHECK(strcmp(path.c_str(), file_list[k].first.c_str()) == 0) << "unmatched compare " << path;
	  std::map<int, int>::iterator itr = class2num.find(label);
	  if (itr == class2num.end())
	  {
		  class2num[label] = 1;
		  class2hit[label] = hit_flag == true;
	  }
	  else
	  {
		  class2num[label] += 1;
		  class2hit[label] += (hit_flag == true);
	  }
  }
  float total = 0, hit = 0;
  std::cout << "recalling..." << std::endl;
  for (int k = 0; k < class2num.size(); k++)
  {
	  std::cout << "class " << k << ":" << (float)(class2hit[k]) / class2num[k] << std::endl;
	  total += class2num[k];
	  hit += class2hit[k];
  }
  std::cout << "in total :" << hit / total << std::endl;
  return 0;

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
