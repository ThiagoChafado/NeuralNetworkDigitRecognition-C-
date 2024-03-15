#include "Network.h"
#include <random>

Network::Network(std::vector<int> sizes) : sizes(sizes)
{

    numLayers = sizes.size();
    biases.resize(numLayers - 1);
    weights.resize(numLayers - 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 1; i < numLayers; ++i)
    {
        biases[i - 1] = Eigen::VectorXd::Random(sizes[i]);
    }

    for (int i = 0; i < numLayers - 1; ++i)
    {
        weights[i] = Eigen::MatrixXd::Random(sizes[i + 1], sizes[i]);
    }
}

std::vector<Eigen::VectorXd> &Network::getBiases()
{
    return biases;
}

std::vector<Eigen::MatrixXd> &Network::getWeights()
{
    return weights;
}

Eigen::VectorXd Network::sigmoid(const Eigen::VectorXd &z)
{
    return 1.0 / (1.0 + (-z.array()).exp());
}

Eigen::VectorXd Network::feedForward(std::vector<Eigen::VectorXd> &biases, std::vector<Eigen::MatrixXd> &weights, Eigen::VectorXd &a)
{
    Eigen::VectorXd result = a;
    for (int i = 0; i < biases.size(); ++i)
    {
        result = sigmoid((weights[i] * result + biases[i]).array());
    }
    return result;
}

void Network::SGD(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> &trainingData, int epochs, int miniBatchSize, double eta, const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> *testData)
{
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> trainingDataCopy = trainingData;
    int n = trainingDataCopy.size();

    int n_test = 0;
    if (testData)
    {
        n_test = testData->size();
    }

    for (int j = 0; j < epochs; ++j)
    {
        std::shuffle(trainingDataCopy.begin(), trainingDataCopy.end(), std::mt19937{std::random_device{}()});

        std::vector<std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>> miniBatches;
        for (size_t k = 0; k < trainingDataCopy.size(); k += miniBatchSize)
        {
            std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> miniBatch(trainingDataCopy.begin() + k, trainingDataCopy.begin() + std::min(k + miniBatchSize, (size_t)n));
            miniBatches.push_back(miniBatch);
        }

        for (const auto &miniBatch : miniBatches)
        {
            update_mini_batch(miniBatch, eta);
        }

        if (testData)
        {
            std::cout << "Epoch " << j << " : " << evaluate(*testData) << " / " << n_test << std::endl;
        }
    }
}

void Network::update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> &miniBatch, double eta)
{
    std::vector<Eigen::MatrixXd> nablaW;
    std::vector<Eigen::VectorXd> nablaB;

    // init gradients
    for (const auto &w : weights)
    {
        nablaW.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
    }

    for (const auto &b : biases)
    {
        nablaB.push_back(Eigen::VectorXd::Zero(b.rows()));
    }

    // gradient calculation for minibatchs
    for (const auto &example : miniBatch)
    {
        const Eigen::VectorXd &x = example.first;
        const Eigen::VectorXd &y = example.second;

        std::vector<Eigen::MatrixXd> deltaNablaW;
        std::vector<Eigen::VectorXd> deltaNablaB;

        std::tie(deltaNablaB, deltaNablaW) = backprop(x, y);

        // Atualização dos gradientes
        for (size_t i = 0; i < nablaW.size(); ++i)
        {
            nablaW[i] += deltaNablaW[i];
            nablaB[i] += deltaNablaB[i];
        }
    }

    // init
    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] -= (eta / miniBatch.size()) * nablaW[i];
        biases[i] -= (eta / miniBatch.size()) * nablaB[i];
    }
}

int Network::evaluate(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> &testData)
{
    int correct = 0;

    for (const auto &example : testData)
    {
        Eigen::VectorXd x = example.first;
        Eigen::VectorXd y = example.second;

        Eigen::VectorXd output = feedForward(biases, weights, x);

        int predictedLabel = findMaxItem(output);
        int trueLabel = findMaxItem(y);

        if (predictedLabel == trueLabel)
        {
            correct++;
        }
        else
        {
            // This prints if the answer is incorrect
            // std::cout << "Predicted: " << predictedLabel << "\n" << "True label: " << trueLabel << "\n";
        }
    }

    return correct;
}

int Network::findMaxItem(const Eigen::VectorXd &vec)
{
    int maxIndex = 0;
    double maxValue = vec(0);

    for (int i = 1; i < vec.size(); ++i)
    {
        if (vec(i) > maxValue)
        {
            maxValue = vec(i);
            maxIndex = i;
        }
    }

    return maxIndex;
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network::backprop(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
{
    std::vector<Eigen::MatrixXd> nablaW;
    std::vector<Eigen::VectorXd> nablaB;

    for (const auto &w : weights)
    {
        nablaW.push_back(Eigen::MatrixXd::Zero(w.rows(), w.cols()));
    }

    for (const auto &b : biases)
    {
        nablaB.push_back(Eigen::VectorXd::Zero(b.rows()));
    }

    // Feedforward
    Eigen::VectorXd activation = x;

    // list for the activations
    std::vector<Eigen::VectorXd> activations;
    activations.push_back(x);

    // list for the z vectors
    std::vector<Eigen::VectorXd> zs;

    for (size_t i = 0; i < biases.size(); ++i)
    {
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        zs.push_back(z);
        activation = sigmoid(z);
        activations.push_back(activation);
    }

    // Backward pass
    Eigen::VectorXd delta = costDerivative(activations.back(), y).array() * sigmoidPrime(zs.back()).array();
    nablaB.back() = delta;
    nablaW.back() = delta * activations[activations.size() - 2].transpose();

    // l = 1 is the last neurons layer, l = 2 is the second, .......
    //
    for (int l = 2; l < numLayers; ++l)
    {
        Eigen::VectorXd z = zs[zs.size() - l];
        Eigen::VectorXd sp = sigmoidPrime(z);
        delta = (weights[weights.size() - l + 1].transpose() * delta).array() * sp.array();
        nablaB[nablaB.size() - l] = delta;
        nablaW[nablaW.size() - l] = delta * activations[activations.size() - l - 1].transpose();
    }

    return std::make_pair(nablaB, nablaW);
}

Eigen::VectorXd Network::sigmoidPrime(const Eigen::VectorXd &z)
{
    return sigmoid(z).array() * (1.0 - sigmoid(z).array());
}

Eigen::VectorXd Network::costDerivative(const Eigen::VectorXd &outputActivations, const Eigen::VectorXd &y)
{
    return outputActivations - y;
}
