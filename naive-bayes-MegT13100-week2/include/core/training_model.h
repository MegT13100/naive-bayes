#pragma once

#include "data.h"
#include "image.h"
#include <fstream>
#include <iostream>
#include <istream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace naivebayes {

class TrainingModel {

public:
  /**
   * TrainingModel constructor that takes in a Data reference.
   * @param data Data reference
   */
  explicit TrainingModel(Data &data);

  /**
   * TrainingModel constructor that takes in an image size.
   * @param image_size size of the image
   */
  TrainingModel(const size_t image_size);

  /**
   * Initializes feature_probability_vector_.
   */
  void InitializeFeatureProbVector();

  /**
    * >> overload.
    *
    * @param is input stream
    * @param training_model TrainingModel reference
    * @return input stream
    */
  friend std::istream& operator>>(std::istream &is, TrainingModel &training_model);

  /**
   * << overload.
   *
   * @param os output stream
   * @param training_model TrainingModel reference
   * @return output stream
   */
  friend std::ostream& operator<<(std::ostream &os, TrainingModel &training_model);

  /**
    * Sets the prior probabilities map.
    */
  void SetPriorProbabilities(Data &data);

  /**
    * Sets the feature probabilities vector.
    */
  void ComputeFeatureProbabilities(Data &data);

  /**
   *
   * @param pixels
   * @return
   */
  int Classification(const vector<vector<size_t>>& pixels);

  /**
   * Underflow calculation.
   *
   * @param data Data reference
   * @param image_number label of the image
   * @return
   */
  double Underflow(const vector<vector<size_t>>& pixels, size_t image_number);

  //Getters
  std::map<size_t, double> GetPriorProbabilities();

  vector<vector<vector<std::map<size_t, double>>>> GetFeatureProbabilities();

  const Data &GetData();

private:
  // Data variable
  Data data_;

  // vectors
  vector<vector<vector<std::map<size_t, double>>>> feature_probability_vector_;
  std::vector<size_t> labels_;

  // maps
  std::map<size_t, double> prior_probability_map_;
  std::map<size_t, int> class_count_;


  // Constant variables
  const double kNumberOfShades = 2.0;
  const double kSmoothingConstant = 1.0;
  const char kSpace = ' ';

  /**
    * Computes the prior probability.
    *
    * @param class_value
    * @return
    */
  double ComputePriorProbabilities(size_t class_number);

  /**
   * Underflow helper.
   *
   * @param data Data reference
   * @param image_number label of the image
   * @return double representing the sum of all the logs of the feature probabilities.
   */
  double UnderflowHelper(const vector<vector<size_t>>& pixels, size_t image_number);

};

}

