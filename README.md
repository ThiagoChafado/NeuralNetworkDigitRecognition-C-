# NeuralNetworkDigitRecognition-C-

This is a project to make a neural network for digit recognition using only maths and c++


For compile the program,you need Eigen3 and opencv4,Eigen3 is a library for linear algebra,opencv4 is a library for open the mnist dataset

How to compile:
g++ -o main *.cpp -I/usr/include/eigen3 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml `pkg-config --cflags --libs opencv4`

In -I/usr/include/eigen3 you need to add the path for te eigen3 library
This only works for linux/ubuntu

Don't know if works on windows...

Execute:
./main

The mnist data set is in the data folder

After you run,it will take a long time before it starts to print the epochs,because eigen3 is not so great like tensorflow to use gpu
