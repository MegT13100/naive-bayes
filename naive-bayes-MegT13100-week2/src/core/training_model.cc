#include <core/file_handler.h>
#include <core/training_model.h>
#include <map>
#include <cmath>
#include <vector>

// rename this class to model

using naivebayes::Pixel;
using std::vector;

namespace naivebayes {

TrainingModel::TrainingModel(Data &data) : data_(data) {
  SetPriorProbabilities(data);

  InitializeFeatureProbVector();
  ComputeFeatureProbabilities(data);
}

TrainingModel::TrainingModel(const size_t image_size) : data_(image_size) {}

void TrainingModel::InitializeFeatureProbVector() {
  for (size_t i = 0; i < data_.GetImageSize(); i++) {
    feature_probability_vector_.emplace_back();
    for (size_t j = 0; j < data_.GetImageSize(); j++) {
      feature_probability_vector_[i].emplace_back();
      for (size_t s = 0; s < kNumberOfShades; s++) {
        feature_probability_vector_[i][j].emplace_back();
      }
    }
  }
}

template <typename T> void TryReadOrThrow(std::istream &is, T &obj) {
  if (!(is >> obj)) {
    throw std::invalid_argument("Insufficient data in file.");
  }
}

std::istream &operator>>(std::istream &is, TrainingModel &training_model) {
  size_t num_of_priors = 0;
  TryReadOrThrow<size_t>(is, num_of_priors);

  size_t key = 0;
  double value = 0;

  for (size_t i = 0; i < num_of_priors; ++i) {
    TryReadOrThrow<size_t>(is, key);
    TryReadOrThrow<double>(is, value);
    training_model.labels_.push_back(key);
    training_model.prior_probability_map_[key] = value;
  }

  // clear
  training_model.feature_probability_vector_.clear();

  // resize before every level except for last level because it's a map
  training_model.feature_probability_vector_.resize(
      training_model.GetData().GetImageSize());
  for (size_t i = 0; i < training_model.GetData().GetImageSize(); i++) {
    training_model.feature_probability_vector_[i].resize(
        training_model.GetData().GetImageSize());
    for (size_t j = 0; j < training_model.GetData().GetImageSize(); j++) {
      training_model.feature_probability_vector_[i][j].resize(
          training_model.kNumberOfShades);
      for (size_t s = 0; s < training_model.kNumberOfShades; s++) {
        for (size_t c = 0; c < num_of_priors; c++) {
          TryReadOrThrow<double>(
              is, training_model.feature_probability_vector_[i][j][s][c]);
        }
      }
    }
  }

  return is;
}

std::ostream &operator<<(std::ostream &os, TrainingModel &training_model) {
  os << training_model.prior_probability_map_.size() << std::endl;
  for (const auto &prob : training_model.prior_probability_map_) {
    os << prob.first << training_model.kSpace << prob.second
       << training_model.kSpace;
  }

  for (size_t i = 0; i < training_model.GetData().GetImageSize(); i++) {
    for (size_t j = 0; j < training_model.GetData().GetImageSize(); j++) {
      for (size_t s = 0; s < training_model.kNumberOfShades; s++) {
        for (const size_t &c:training_model.GetData().GetLabels()) {
          float value = training_model.feature_probability_vector_[i][j][s][c];
          os << value << training_model.kSpace;
        }
      }
    }
  }
  return os;
}

void TrainingModel::SetPriorProbabilities(Data &data) {
  for (const Image &img : data.GetImages()) {
    if(class_count_.count(img.GetLabel())==0){
      class_count_[img.GetLabel()] = 0;
    }
    class_count_[img.GetLabel()]++;
  }
  for (const auto &label : data.GetLabels()) {
    prior_probability_map_.insert(
        std::pair<size_t, double>(label, ComputePriorProbabilities(label)));
  }
}

double TrainingModel::ComputePriorProbabilities(size_t class_number) {
  double numerator =
      kSmoothingConstant + class_count_[class_number];
  size_t denominator = (data_.GetLabels().size() * kSmoothingConstant) +
                       data_.GetImages().size();

  return (double)numerator / denominator;
}

void TrainingModel::ComputeFeatureProbabilities(Data &data) {
  for (size_t i = 0; i < data.GetImageSize(); i++) {
    for (size_t j = 0; j < data.GetImageSize(); j++) {
      for (const Image &img : data.GetImages()) {
        feature_probability_vector_[i][j][img.GetImage()[i][j]]
                                   [img.GetLabel()]++;
      }
      // compute feature
      for (size_t s = 0; s < kNumberOfShades; s++) {
        for (const size_t &label : data.GetLabels()) {
          feature_probability_vector_[i][j][s][label] += kSmoothingConstant;
          feature_probability_vector_[i][j][s][label] /=
              data.GetLabels().size() * kSmoothingConstant +
              class_count_[label];
        }
      }
    }
  }
}

// Classify
int TrainingModel::Classification(const vector<vector<size_t>> &pixels) {
  double highest_likelihood = -std::numeric_limits<double>::max();
  int most_likely = -1;
  for (const size_t& label : labels_) { //data.GetLabels() or labels_
    double score = Underflow(pixels, label);
    if (score > highest_likelihood) {
      highest_likelihood = score;
      most_likely = label;
    }
  }
  return most_likely;
}

// Math
double TrainingModel::Underflow(const vector<vector<size_t>> &pixels,
                                size_t class_number) {
  double prior_probability = log(prior_probability_map_[class_number]);
  double feature_probability = UnderflowHelper(pixels, class_number);

  std::cout << prior_probability + feature_probability;
  return prior_probability + feature_probability;
}

double TrainingModel::UnderflowHelper(const vector<vector<size_t>> &pixels,
                                      size_t class_number) {
  double total_log_sum = 0.0;
  for (size_t i = 0; i < data_.GetImageSize(); i++) {
    for (size_t j = 0; j < data_.GetImageSize(); j++) {
      auto tmp =
          // error is here, feature is not set correctly
          log(feature_probability_vector_[i][j][pixels[i][j]][class_number]);
      total_log_sum += tmp;
    }
  }
  return total_log_sum;
}

// Getters
std::map<size_t, double> TrainingModel::GetPriorProbabilities() {
  return prior_probability_map_;
}

vector<vector<vector<std::map<size_t, double>>>>
TrainingModel::GetFeatureProbabilities() {
  return feature_probability_vector_;
}

const Data &TrainingModel::GetData() { return data_; }


} // namespace naivebayes