#include <core/data.h>
#include <core/training_model.h>
#include <core/image.h>

// TODO: You may want to change main's signature to take in argc and argv
//
int main() {
  // Create trainer and data
  // >> (load)
  naivebayes::Data data(28);
  std::ifstream input_file("../data/trainingimagesandlabels.txt");
  input_file>>data;
  naivebayes::TrainingModel trainer(data);

  // << (save)
  std::ofstream save_file;
  save_file.open("../data/outstream_file.txt");
  save_file<<trainer;

}
