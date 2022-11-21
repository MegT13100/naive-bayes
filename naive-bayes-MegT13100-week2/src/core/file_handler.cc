#include "core/file_handler.h"

namespace naivebayes {

/**
 * FileHandler constructor.
 * @param file_path path of the file
 * @param training_model TrainingModel reference
 */
FileHandler::FileHandler(std::string file_path, Data &data)
    : file_path_(file_path), data_(data) {}

// Getters
std::string FileHandler::GetFilePath() {return file_path_;}
Data FileHandler::GetData() {return data_;}

/**
 * Handles reading the file.
 */
void FileHandler::HandleFile() {
  if (IsFileValid(file_path_)) {
    throw std::exception();
  }

  std::ifstream if_(file_path_);
  if_ >> data_;
}

/**
 * Determines of the file is valid.
 * @param file_path path of the file
 * @return true if the file is valid, false otherwise
 */
bool FileHandler::IsFileValid(std::string file_path) {
  // test if actual file is empty
  int number_of_lines = 0;
  std::string line;
  std::ifstream my_file(file_path);

  while (std::getline(my_file, line)) ++number_of_lines;

  if (number_of_lines <= 1) {
    throw std::invalid_argument("file is empty");
  }

  // tests if path is empty
  if (file_path.empty()) {
    throw std::invalid_argument("empty file_path");
  }

  // other tests
  std::ifstream pFile(file_path);
  return pFile.peek() == std::ifstream::traits_type::eof();
}

}