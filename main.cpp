#include "Network.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);
    std::cout << static_cast<int>(numImagesBytes[0]) << "  " << static_cast<int>(numImagesBytes[1]) << "  " << (int)static_cast<unsigned char>(numImagesBytes[2]) << "  " << static_cast<int>(numImagesBytes[3]) << "  " << std::endl;

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | (static_cast<unsigned char>(numRowsBytes[1]) << 16) | (static_cast<unsigned char>(numRowsBytes[2]) << 8) | static_cast<unsigned char>(numRowsBytes[3]);
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | (static_cast<unsigned char>(numColsBytes[1]) << 16) | (static_cast<unsigned char>(numColsBytes[2]) << 8) | static_cast<unsigned char>(numColsBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++)
    {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(numRows * numCols);
        file.read((char *)(image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}

// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>> readLabelFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++)
    {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(1);
        file.read((char *)(image.data()), 1);

        images.push_back(image);
    }

    file.close();

    return images;
}

int main()
{
    // Leia os dados MNIST
    std::string filename = "/home/thiagoalmeida/Desktop/Projects/NeuralDigitRecognition/data/train-images-idx3-ubyte/train-images.idx3-ubyte";
    std::string label_filename = "/home/thiagoalmeida/Desktop/Projects/NeuralDigitRecognition/data/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
    std::vector<std::vector<unsigned char>> imagesFile = readIDX3UByteFile(filename);
    std::vector<std::vector<unsigned char>> labelsFile = readLabelFile(label_filename);

    // converting Mnist to Eigen
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
    for (size_t i = 0; i < imagesFile.size(); ++i)
    {
        Eigen::VectorXd imageVec = Eigen::VectorXd::Zero(imagesFile[i].size());
        for (size_t j = 0; j < imagesFile[i].size(); ++j)
        {
            imageVec(j) = static_cast<double>(imagesFile[i][j]) / 255.0; // Normalizing vectors
        }
        Eigen::VectorXd labelVec = Eigen::VectorXd::Zero(10); 
        labelVec(labelsFile[i][0]) = 1.0;                     
        training_data.emplace_back(std::make_pair(imageVec, labelVec));
    }

    // 80% training,20% test
    std::random_shuffle(training_data.begin(), training_data.end());
    size_t split_index = static_cast<size_t>(0.8 * training_data.size());
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_set(training_data.begin(), training_data.begin() + split_index);
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> test_set(training_data.begin() + split_index, training_data.end());

    std::vector<int> sizes = {28 * 28, 30, 10}; // 28x28 pixels, 30 nodes on hidden layer, 10 nodes out

    Network network(sizes);

    // Training
    int epochs = 30;
    int mini_batch_size = 10;
    double eta = 3.0;
    network.SGD(training_set, epochs, mini_batch_size, eta, &test_set);
    

    return 0;
}
