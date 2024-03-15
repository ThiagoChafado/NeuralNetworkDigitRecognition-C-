#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include "iostream"
#include <cmath>
#include <random>
class Network
{
public:
    Network(std::vector<int> sizes);

    // Returns the value of the sigmoid function f(x) = 1/(1 + e^-x)
    Eigen::VectorXd sigmoid(const Eigen::VectorXd &z);

    // Apply the sigmoid function for each layer
    Eigen::VectorXd feedForward(std::vector<Eigen::VectorXd> &biases, std::vector<Eigen::MatrixXd> &weights, Eigen::VectorXd &a);
    std::vector<Eigen::VectorXd> &getBiases();
    std::vector<Eigen::MatrixXd> &getWeights();

    void update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> &miniBatch, double eta);
    int evaluate(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> &testData);
    int findMaxItem(const Eigen::VectorXd &vec);
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop(const Eigen::VectorXd &x, const Eigen::VectorXd &y);
    Eigen::VectorXd sigmoidPrime(const Eigen::VectorXd &z);
    void SGD(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> &trainingData, int epochs, int miniBatchSize, double eta, const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> *testData);
    Eigen::VectorXd costDerivative(const Eigen::VectorXd& outputActivations, const Eigen::VectorXd& y);
private:
    int numLayers;
    std::vector<int> sizes;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> weights;
};

#endif // NETWORK_H