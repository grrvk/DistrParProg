#include <iostream>
#include <filesystem>
#include <vector>
#include <mpi.h>
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double globalStart = MPI_Wtime();

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::filesystem::path inputDir = argv[1];
    std::filesystem::path outputDir = argv[2];
    int max_images = atoi(argv[3]);

    if (world_rank == 0) {
        cout << "Process running with:\n- input_dir: " << inputDir.generic_string() << endl;
        cout << "- output_dir: " << outputDir.generic_string() << "\n- num_of_processes: " << world_size << endl;
        cout << "- max_images: " << max_images << endl << endl;
    }

    vector<string> fileList;
    if (world_rank == 0) {
        if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir)) {
            cout << "Error: Input directory does not exist or is not a directory." << endl;
            MPI_Abort(MPI_COMM_WORLD, 0);
        }

        for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
            std::filesystem::path filePath = entry.path();
            if (filePath.extension() == ".png" || filePath.extension() == ".jpg") {
                fileList.push_back(filePath.generic_string());
            }
        }
    
        if (std::filesystem::exists(outputDir)) {
            for (const auto& entry : std::filesystem::directory_iterator(outputDir)) {
                std::filesystem::remove_all(entry.path());
            }
        } else {
            std::filesystem::create_directories(outputDir);
        }
    }

    int numFiles = fileList.size();
    if (numFiles > max_images) {
        fileList.resize(max_images);
        numFiles = max_images;
    }

    MPI_Bcast(&numFiles, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int MAX_NAME_SIZE = 100;
    char fileNames[numFiles][MAX_NAME_SIZE];
    if (world_rank == 0) {
        for (int i = 0; i < numFiles; i++) {
            std::strncpy(fileNames[i], fileList[i].c_str(), MAX_NAME_SIZE);
        }
    }

    MPI_Bcast(fileNames, numFiles * MAX_NAME_SIZE, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        fileList.clear();
        for (int i = 0; i < numFiles; i++) {
            fileList.push_back(std::string(fileNames[i]));
        }
    }

    double localSharpenTime = 0;
    int localFileCount = 0;
    for (int i = world_rank; i < numFiles; i += world_size) {
        string filePathStr = fileList[i];
        std::filesystem::path filePath(filePathStr);
        
        cv::Mat inputImage = cv::imread(filePath);
        if (inputImage.empty()) {
            cout << "Error: Could not load image." << endl;
            MPI_Abort(MPI_COMM_WORLD, i);
        }
        
        double localSharpenIterationStart = MPI_Wtime();
        cv::Mat outputImage;
        applySharpeningFilter(inputImage, outputImage);
        double localSharpenIterationEnd = MPI_Wtime();
        double processingTime = localSharpenIterationEnd - localSharpenIterationStart;

        std::filesystem::path outputFilePath = outputDir / filePath.filename();

        if (!cv::imwrite(outputFilePath.string(), outputImage)) {
            cout << "Error: Could not write image to " << outputFilePath << std::endl;
            MPI_Abort(MPI_COMM_WORLD, i);
        }
        
        localSharpenTime += processingTime;
        localFileCount++;
    }

    double totalSharpenTime = 0;
    int totalFileCount = 0;
    
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Reduce(&localSharpenTime, &totalSharpenTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localFileCount, &totalFileCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double globalEnd = MPI_Wtime();

    if (world_rank == 0) {
        double globalTimePassed = globalEnd - globalStart;
        cout << "Processed " << totalFileCount << " files:" << endl;
        cout << "- total wall clock runtime: " << globalTimePassed << endl;
        cout << "- total sharpen time: " << totalSharpenTime << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    cout << "  - rank " << world_rank << " sharpen time: " << localSharpenTime << endl;

    MPI_Finalize();
    return 0;
}