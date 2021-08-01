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

ArmorClassifier::ArmorClassifier(const std::string model_path,
                                 const std::string lable_path,
                                 const cv::Size &input_size) {
  LoadModel(model_path);
  LoadLable(lable_path);
  SetInputSize(input_size);
  SPDLOG_TRACE("Constructed.");
}

ArmorClassifier::ArmorClassifier() { SPDLOG_TRACE("Constructed."); }
ArmorClassifier::~ArmorClassifier() { SPDLOG_TRACE("Destructed."); }

void ArmorClassifier::LoadModel(const std::string &path) {
  net_ = cv::dnn::readNet(path);
}

void ArmorClassifier::LoadLable(const std::string &path) {
  cv::FileStorage fs(path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  cv::FileNode root = fs.root();
  for (size_t i = 0; i < root.size(); ++i) {
    classes_.push_back(
        game::StringToModel(std::string(root[std::to_string(i)])));
  }
}

void ArmorClassifier::SetInputSize(const cv::Size &input_size) {
  net_input_size_ = input_size;
}

void ArmorClassifier::ClassifyModel(Armor &armor, const cv::Mat &frame) {
  cv::Mat image = armor.ImageFace(frame);
  cv::dnn::blobFromImage(image, blob_, 1. / 128., net_input_size_);
  net_.setInput(blob_);
  cv::Mat prob = net_.forward();
  cv::Point class_point;
  cv::minMaxLoc(prob.reshape(1, 1), nullptr, &conf_, nullptr, &class_point);
  model_ = classes_[class_point.x];
  armor.SetModel(model_);
}
