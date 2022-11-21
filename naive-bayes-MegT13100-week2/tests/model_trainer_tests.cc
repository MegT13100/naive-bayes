#include <catch2/catch.hpp>

#include <core/image.h>
#include <core/training_model.h>
#include <core/file_handler.h>

TEST_CASE("Correct Size") {
  SECTION("Size 3") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetImages().size() == 3);
  }

  SECTION("Size 5") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetImages().size() == 5);
  }

  SECTION("Size 28 one image") {
    naivebayes::Data data(28);
    std::ifstream input_file("../../../../../../data/large_numbers_one.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetImages().size() == 1);
  }

  SECTION("Giant data file ") {
    naivebayes::Data data(28);
    std::ifstream input_file("../../../../../../data/trainingimagesandlabels.txt");
    input_file>>data;
    REQUIRE(data.GetImages().size() == 5000);
  }

}

TEST_CASE("Test Correct Label Values") {
  SECTION("Size 3 Dataset") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetImages().size() == 3);
    REQUIRE(trainer.GetData().GetImages().at(0).GetLabel() == 0);
    REQUIRE(trainer.GetData().GetImages().at(1).GetLabel() == 1);
    REQUIRE(trainer.GetData().GetImages().at(2).GetLabel() == 4);
  }

  SECTION("Size 5 Dataset") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetImages().size() == 5);
    REQUIRE(trainer.GetData().GetImages().at(0).GetLabel() == 0);
    REQUIRE(trainer.GetData().GetImages().at(1).GetLabel() == 1);
    REQUIRE(trainer.GetData().GetImages().at(2).GetLabel() == 4);
    REQUIRE(trainer.GetData().GetImages().at(3).GetLabel() == 7);
    REQUIRE(trainer.GetData().GetImages().at(4).GetLabel() == 5);
  }
}

TEST_CASE("Test Correct Labels Size") {
  SECTION("Non repeating labels Size 3") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetLabels().size() == 3);
  }

  SECTION("Non repeating labels Size 5") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetLabels().size() == 5);
  }

  SECTION("Repeating labels Size 3") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetLabels().size() == 3);
  }

  SECTION("Repeating labels Size 5") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    REQUIRE(trainer.GetData().GetLabels().size() == 5);
  }

}

TEST_CASE("Test File Reader") {
  SECTION("Size 3 Dataset") {
    naivebayes::Data data(3);
    data.FileReader("../../../../../../data/size_three_no_repeats.txt", data);

    REQUIRE(data.GetImages().size() == 3);
    REQUIRE(data.GetImages().at(0).GetLabel() == 0);
    REQUIRE(data.GetImages().at(1).GetLabel() == 1);
    REQUIRE(data.GetImages().at(2).GetLabel() == 4);
  }

  SECTION("Size 5 Dataset") {
    naivebayes::Data data(5);
    data.FileReader("../../../../../../data/size_five_no_repeats.txt", data);

    REQUIRE(data.GetImages().size() == 5);
    REQUIRE(data.GetImages().at(0).GetLabel() == 0);
    REQUIRE(data.GetImages().at(1).GetLabel() == 1);
    REQUIRE(data.GetImages().at(2).GetLabel() == 4);
    REQUIRE(data.GetImages().at(3).GetLabel() == 7);
    REQUIRE(data.GetImages().at(4).GetLabel() == 5);
  }

  SECTION("Test empty file") {
    naivebayes::Data data(5);
    naivebayes::FileHandler file_handler("../../../../../../data/empty_file.txt", data);

    REQUIRE_THROWS_AS(file_handler.HandleFile(), std::invalid_argument);
  }

  SECTION("Test null file_path") {
    naivebayes::Data data(2);
    std::string null_file;
    naivebayes::FileHandler file_handler(null_file, data);

    REQUIRE_THROWS_AS(file_handler.HandleFile(), std::invalid_argument);
  }

}

TEST_CASE("Prior Probability Tests") {
  SECTION("Size 3 no repeat images") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 3);
    REQUIRE(trainer.GetPriorProbabilities().at(0) == Approx(0.3333333333).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(1) == Approx(0.3333333333).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(4) == Approx(0.3333333333).epsilon(0.01));
  }

  SECTION("Size 3 with repeat images") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 3);
    REQUIRE(trainer.GetPriorProbabilities().at(0) == Approx(0.4285714286).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(1) == Approx(0.2857142857).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(4) == Approx(0.2857142857).epsilon(0.01));
  }

  SECTION("Size 5 no repeat images") {
    // image instance with the size
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 5);
    REQUIRE(trainer.GetPriorProbabilities().at(0) == Approx(0.2).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(1) == Approx(0.2).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(4) == Approx(0.2).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(5) == Approx(0.2).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(7) == Approx(0.2).epsilon(0.01));
  }

  SECTION("Size 5 with repeat images") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 5);
    REQUIRE(trainer.GetPriorProbabilities().at(0) == Approx(0.2727272727).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(1) == Approx(0.1818181818).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(4) == Approx(0.1818181818).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(5) == Approx(0.1818181818).epsilon(0.01));
    REQUIRE(trainer.GetPriorProbabilities().at(7) == Approx(0.1818181818).epsilon(0.01));
  }
}

TEST_CASE("Pixel Tests") {
  SECTION("Shaded or Unshaded Correctly") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    // pixel for 0
    std::vector<std::vector<size_t>> pixels {
        {Pixel::kShadedPixel, Pixel::kShadedPixel, Pixel::kShadedPixel},
        {Pixel::kShadedPixel, Pixel::kUnshadedPixel, Pixel::kShadedPixel},
        {Pixel::kShadedPixel, Pixel::kShadedPixel, Pixel::kShadedPixel}
    };

    REQUIRE(data.GetImages().at(0).GetImage() == pixels);
  }

  SECTION("Size 3 no repeat images") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 3);

    REQUIRE(trainer.GetFeatureProbabilities().size() == 3); // rows
    REQUIRE(trainer.GetFeatureProbabilities()[0].size() == 3); // columns
    REQUIRE(trainer.GetFeatureProbabilities()[0][0].size() == 2); // shades
    REQUIRE(trainer.GetFeatureProbabilities()[0][0][0].size() == 3); // labels
  }

  SECTION("Pixel Tests") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);
    // first pixel
    REQUIRE(trainer.GetFeatureProbabilities()[0][0][0][0] == Approx(0.25).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[0][0][1][0] == Approx(0.5).epsilon(0.01));
    // second pixel
    REQUIRE(trainer.GetFeatureProbabilities()[0][1][0][0] == Approx(0.25).epsilon(0.01));
    // third pixel
    REQUIRE(trainer.GetFeatureProbabilities()[1][1][0][0] == Approx(0.5).epsilon(0.01));
    // fourth pixel
    REQUIRE(trainer.GetFeatureProbabilities()[1][2][0][0] == Approx(0.25).epsilon(0.01));
    // fifth pixel
    REQUIRE(trainer.GetFeatureProbabilities()[2][2][0][0] == Approx(0.25).epsilon(0.01));
  }
}

TEST_CASE("Feature Probabilities") {
  SECTION("Size 3 no repeat images") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 3);
    REQUIRE(trainer.GetFeatureProbabilities()[0][0][0][1] == Approx(0.5).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[1][1][1][1] == Approx(0.5).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[2][1][1][1] == Approx(0.5).epsilon(0.01));
  }

  SECTION("Size 3 with repeat images") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 3);
    REQUIRE(trainer.GetFeatureProbabilities()[0][0][0][0] == Approx(0.2).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[1][0][1][1] == Approx(0.25).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[2][1][1][1] == Approx(0.5).epsilon(0.01));
  }

  SECTION("Size 5 no repeat images") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_no_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 5);
    REQUIRE(trainer.GetFeatureProbabilities()[0][0][0][0] == Approx(0.1666666667).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[1][1][1][1] == Approx(0.3333333333).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[2][1][1][1] == Approx(0.3333333333).epsilon(0.01));
  }

  SECTION("Size 5 with repeat images") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.GetData().GetLabels().size() == 5);
    REQUIRE(trainer.GetFeatureProbabilities().size() == 5);
    REQUIRE(trainer.GetFeatureProbabilities()[0][0][0][0] == Approx(0.1428571429).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[1][1][1][1] == Approx(0.3333333333).epsilon(0.01));
    REQUIRE(trainer.GetFeatureProbabilities()[2][1][1][1] == Approx(0.3333333333).epsilon(0.01));
  }
}

// underflow
TEST_CASE("Underflow Equation") {
  SECTION("Size 3 no repeats") {
    naivebayes::Data data(3);
    std::ifstream input_file("../../../../../../data/size_three_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.Classification(data.GetImages().at(0).GetImage()) ==
            Approx(-1).epsilon(.01));
    REQUIRE(trainer.Classification(data.GetImages().at(1).GetImage()) ==
            Approx(-1).epsilon(.01));
    REQUIRE(trainer.Classification(data.GetImages().at(2).GetImage()) ==
            Approx(-1).epsilon(.01));
  }

  SECTION("Size 5 with repeats") {
    naivebayes::Data data(5);
    std::ifstream input_file("../../../../../../data/size_five_with_repeats.txt");
    input_file>>data;
    naivebayes::TrainingModel trainer(data);

    REQUIRE(trainer.Classification(data.GetImages().at(0).GetImage()) == -1);

    /*
    REQUIRE(trainer.Underflow(data.GetImages().at(0).GetImage(), 0) ==
            Approx(-22.4817294938).epsilon(.01));
    REQUIRE(trainer.Underflow(data.GetImages().at(1).GetImage(), 1) ==
            Approx(-29.1700553089).epsilon(.01));
    REQUIRE(trainer.Underflow(data.GetImages().at(2).GetImage(), 4) ==
            Approx(-29.1700553089).epsilon(.01));
    REQUIRE(trainer.Underflow(data.GetImages().at(3).GetImage(), 5) ==
            Approx(-34.7152327534).epsilon(.01));
    REQUIRE(trainer.Underflow(data.GetImages().at(4).GetImage(), 7) ==
            Approx(-34.7152327534).epsilon(.01));
    */
  }
}

// accuracy test
TEST_CASE("Accuracy above 70%") {
  naivebayes::Data data(28);
  std::ifstream input_file("../../../../../../data/trainingimagesandlabels.txt");
  input_file>>data;
  naivebayes::TrainingModel trainer(data);

  naivebayes::Data test_data(28);
  std::ifstream test_file("../../../../../../data/testimagesandlabels.txt");
  test_file >> test_data;

  double accuracy = 0.0;
  for (const naivebayes::Image& img : test_data.GetImages()) {
    if (trainer.Classification(
            (const vector<vector<size_t>> &)img.GetImage()) == img.GetLabel()) {
      accuracy++;
    }
  }

  accuracy /= data.GetImages().size();

  REQUIRE(accuracy > 0.7);
}



