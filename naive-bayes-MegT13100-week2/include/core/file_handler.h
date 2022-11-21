#include <string>
#include <vector>
#include <core/training_model.h>

namespace naivebayes {

class FileHandler{
public:
  /**
    * FileHandler constructor.
    * @param file_path path of the file
    * @param training_model TrainingModel reference
    */
  FileHandler(std::string file_path, Data &data);

  // Getters
  std::string GetFilePath();
  Data GetData();

  /**
    * Handles reading the file.
    */
  void HandleFile();

  /**
    * Determines of the file is valid.
    * @param file_path path of the file
    * @return true if the file is valid, false otherwise
    */
  bool IsFileValid(std::string file_path);

private:
  // private variables
  std::string file_path_;
  Data &data_;

};


}



