#include "armor_classifier.hpp"

#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);

}  // namespace

ArmorClassifier::ArmorClassifier(const std::string model_path, int width,
                                 int height) {
  LoadModel(model_path);
  SetInputSize(width, height);
  SPDLOG_TRACE("Constructed.");
}

ArmorClassifier::ArmorClassifier() { SPDLOG_TRACE("Constructed."); }
ArmorClassifier::~ArmorClassifier() { SPDLOG_TRACE("Destructed."); }

void ArmorClassifier::LoadModel(const std::string &path) {
  net_ = cv::dnn::readNet(path);
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void ArmorClassifier::SetInputSize(int width, int height) {
  net_input_size_ = cv::Size(width, height);
}

void ArmorClassifier::ClassifyModel(Armor &armor) {
  cv::Mat frame = armor.Face(frame);
  cv::dnn::blobFromImage(frame, blob_, scale_, net_input_size_, mean_, true,
                         false);
  net_.setInput(blob_);
  cv::Mat prob = net_.forward();
  cv::Point class_point;
  cv::minMaxLoc(prob.reshape(1, 1), nullptr, &conf_, nullptr, &class_point);
  model_ = classes_[class_point.x];
  armor.SetModel(model_);
}

void ArmorClassifier::VisualizeResult(const cv::Mat &output, int verbose) {
  std::vector<double> layers_times;
  double freq = cv::getTickFrequency() / 1000.;
  double t = net_.getPerfProfile(layers_times) / freq;
  std::string label = cv::format("%.2f ms, %s: %.4f", t,
                                 game::ModelToString(model_).c_str(), conf_);
  cv::putText(output, label, cv::Point(0, 0), kCV_FONT, 1., kGREEN);
}
