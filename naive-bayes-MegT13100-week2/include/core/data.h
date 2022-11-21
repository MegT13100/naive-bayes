#pragma once
#include <vector>
#include "image.h"

// should have the first overload.

// this should pass into the training_model

namespace naivebayes {

class Data {

public:
  /**
   * Data constructor.
   * @param image_size size of the image
   */
  explicit Data(size_t image_size);

  /**
   * >> overload.
   *
   * @param is input stream
   * @param data Data reference
   * @return input stream
   */
  friend std::istream& operator>>(std::istream &is, Data &data);

  /**
   * Reads the file path and determines if it's valid.
   *
   * @param file_path path to the file
   * @param data Data reference
   */
  void FileReader(std::string file_path, Data &data);

  //Getters
  const vector<Image> &GetImages() const;
  size_t GetImageSize() const;
  const vector<size_t> &GetLabels() const;


private:
  // class variables
  size_t image_size_;

  // class vectors
  vector<Image> images_;
  vector<size_t> labels_;

  // char vector
  const vector<char> shaded_values {'#', '+'};

  /**
   * Keeps track of the Unique labels present in the text file.
   *
   * @param label
   */
  void UpdateAmountOfLabels(size_t label);

  /**
   * Sets the pixel vector which is passed into the Image constructor to
   * create an image.
   *
   * @param pixels String vector
   * @return a Pixel vector
   */
  vector<vector<size_t>> SetPixelVectorForImage(vector<std::string>& pixels) const;
};

}

