#include <core/data.h>
#include <core/file_handler.h>
#include <fstream>

using naivebayes::Pixel;
using naivebayes::Pixel;

namespace naivebayes {

Data::Data(size_t image_size) {
  image_size_ = image_size;
}

std::istream &operator>>(std::istream &is, Data &data) {
  std::string str;
  while(std::getline(is, str)) {

    size_t label = (std::stoi(str));
    data.UpdateAmountOfLabels(label);

    vector<std::string> pixels;
    for (size_t i = 0; i < data.image_size_; i++) {
      std::getline(is, str);
      pixels.push_back(str);
    }
    auto image_shades = data.SetPixelVectorForImage(pixels);
    Image img = Image(label, data.image_size_, image_shades);
    data.images_.emplace_back(img);
  }

  return is;
}

void Data::FileReader(std::string file_path, Data &data) {
  FileHandler file_handler(file_path, data);
  file_handler.HandleFile();
}

void Data::UpdateAmountOfLabels(size_t label) {
  if (std::count(labels_.begin(), labels_.end(), label) == 0) {
    labels_.push_back(label);
  }
}

vector<vector<size_t>> Data::SetPixelVectorForImage(vector<std::string> &pixels) const {
  vector<vector<size_t>> image_shades(image_size_, vector<size_t>(image_size_));

  for (size_t i = 0; i < image_size_; i++) {
    for (size_t j = 0; j < image_size_; j++) {
      if (std::count(shaded_values.begin(), shaded_values.end(), pixels[i][j])) {
        image_shades[i][j] = kShadedPixel;
      } else {
        image_shades[i][j] = kUnshadedPixel;
      }
    }
  }

  return image_shades;
}

// Getters
const vector<Image> &Data::GetImages() const { return images_; }

const vector<size_t> &Data::GetLabels() const { return labels_; }

size_t Data::GetImageSize() const { return image_size_; }

}


