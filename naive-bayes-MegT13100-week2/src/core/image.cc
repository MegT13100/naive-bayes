#include "core/image.h"
using std::vector;
using naivebayes::Pixel;

namespace naivebayes {

Image::Image(size_t label, size_t image_size, vector<vector<size_t>> &image)
    : label_(label), image_size_(image_size), image_(std::move(image)){}

// Getters
const size_t &Image::GetLabel() const {return label_;}

const size_t &Image::GetImageSize() const {return image_size_;}

const vector<vector<size_t>> &Image::GetImage() const {return image_;}


} // namespace naivebayes
