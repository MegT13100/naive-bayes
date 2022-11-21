#pragma once
#include <string>
#include <vector>
#include <core/pixel.h>

using std::vector;
using naivebayes::Pixel;

namespace naivebayes {

class Image {

public:
  /**
   * Image constructor.
   * @param label class number
   * @param image_size size of image
   * @param image pixel vector
   */
  Image(size_t label, size_t image_size, vector<vector<size_t>> &image);

  // Getters
  const size_t &GetLabel() const;
  const size_t &GetImageSize() const;
  const vector<vector<size_t>> &GetImage() const;

private:
  // Private variables
  size_t label_;
  size_t image_size_;
  vector<vector<size_t>> image_;

};


}


