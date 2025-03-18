#include <iostream>
#include <filesystem>
#include <vector>
#include <omp.h>
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
    int num_of_threads = atoi(argv[3]);
    int max_images = atoi(argv[4]);

    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir)) {
        cout << "Error: Input directory does not exist or is not a directory." << endl;
        return 0;
    }

    if (std::filesystem::exists(outputDir)) {
        for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
            std::filesystem::remove_all(entry.path());
        }
    } else {
        std::filesystem::create_directories(outputDir);
    }

    omp_set_num_threads(num_of_threads);
    int num_threads = omp_get_max_threads();
    cout << "Process running with:\n- input_dir: " << inputDir.generic_string() << endl;
    cout << "- output_dir: " << outputDir.generic_string() << "\n- num_of_threads: " << num_threads << endl;
    cout << "- max_images: " << max_images << endl << endl;

    double sharpenTime = 0;
    int fileCount = 0;

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

    double totalTimeStart = omp_get_wtime();

    double threadSharpenTime[num_threads];
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        threadSharpenTime[thread_id] = 0.0;

        #pragma omp for schedule(dynamic) reduction(+:sharpenTime, fileCount)
        for (int i = 0; i < fileList.size(); ++i) {
            const std::filesystem::path& filePath = fileList[i];

            cv::Mat inputImage = cv::imread(filePath);
            if (inputImage.empty()) {
                std::cerr << "Error: Could not load image." << std::endl;
            }

            double sharpenIterationStart = omp_get_wtime();
            cv::Mat outputImage;
            applySharpeningFilter(inputImage, outputImage);
            double sharpenIterationEnd = omp_get_wtime();
            double processingTime = sharpenIterationEnd - sharpenIterationStart;

            std::filesystem::path outputFilePath = outputDir / filePath.filename();

            if (!cv::imwrite(outputFilePath.string(), outputImage)) {
                std::cerr << "Error: Could not write image to " << outputFilePath << std::endl;
            }

            sharpenTime += processingTime;
            fileCount++;
            threadSharpenTime[thread_id] += processingTime;
        }
    }

    double totalTimeEnd = omp_get_wtime();
    double totalTime = totalTimeEnd - totalTimeStart;

    cout << "Processed " << fileCount << " files:" << endl;
    cout << "- total runtime: " << totalTime << endl;
    cout << "- total sharpen time: " << sharpenTime << endl;
    for (int i = 0; i < num_threads; ++i) {
        cout << "  - thread " << i << " sharpen time: " << threadSharpenTime[i] << endl;
    }
    return 0;
}
