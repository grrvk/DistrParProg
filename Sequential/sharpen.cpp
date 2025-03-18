#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;


void applySharpeningFilter(const cv::Mat& inputImage, cv::Mat& outputImage) {
    int height = inputImage.rows;
    int width = inputImage.cols;
    int channels = inputImage.channels();

    outputImage = inputImage.clone();

    const int sharpenKernel[3][3] = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };

    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            for (int c = 0; c < channels; ++c) {
                int sum = 0;

                for (int ki = -1; ki <= 1; ++ki) {
                    for (int kj = -1; kj <= 1; ++kj) {
                        sum += inputImage.at<cv::Vec3b>(i + ki, j + kj)[c] * sharpenKernel[ki + 1][kj + 1];
                    }
                }

                outputImage.at<cv::Vec3b>(i, j)[c] = std::max(0, std::min(255, sum));
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::filesystem::path inputDir = argv[1];
    std::filesystem::path outputDir = argv[2];
    int max_images = atoi(argv[3]);

    cout << "Squential process running with:\n- input_dir: " << inputDir.generic_string() << endl;
    cout << "- output_dir: " << outputDir.generic_string() << endl;
    cout << "- max_images: " << max_images << endl << endl;

    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir)) {
        cout << "Error: Input directory does not exist or is not a directory." << endl;
        return 1;
    }

    if (std::filesystem::exists(outputDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
            std::filesystem::remove_all(entry.path());
        }
    } else {
        std::filesystem::create_directories(outputDir);
    }

    vector<std::filesystem::path> fileList;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        std::filesystem::path filePath = entry.path();
        if (filePath.extension() == ".png" || filePath.extension() == ".jpg") {
            fileList.push_back(filePath);
        }
    }

    if (fileList.size() > max_images) {
        fileList.resize(max_images);
    }

    double totalTimeStart = static_cast<double>(cv::getTickCount());

    int fileCount = 0;
    double sharpenTime = 0;

    for (const auto& filePath : fileList) {
        cv::Mat inputImage = cv::imread(filePath);
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load image: " << filePath << std::endl;
            continue;
        }

        double sharpenIterationStart = static_cast<double>(cv::getTickCount());
        cv::Mat outputImage;
        applySharpeningFilter(inputImage, outputImage);
        double sharpenIterationEnd = static_cast<double>(cv::getTickCount());
        
        sharpenTime += (sharpenIterationEnd - sharpenIterationStart) / cv::getTickFrequency();

        std::filesystem::path outputFilePath = outputDir / filePath.filename();
        if (!cv::imwrite(outputFilePath.string(), outputImage)) {
            std::cerr << "Error: Could not write image to " << outputFilePath << std::endl;
        }

        fileCount++;
    }

    double totalTimeEnd = static_cast<double>(cv::getTickCount());
    double totalTime = (totalTimeEnd - totalTimeStart) / cv::getTickFrequency();

    cout << "Processed " << fileCount << " files:" << endl;
    cout << "- Total runtime: " << totalTime << " sec" << endl;
    cout << "- Total sharpen time: " << sharpenTime << " sec" << endl;

    return 0;
}
