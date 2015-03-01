ffwdnet
=======

The `ffwdnet` application implements a feed-forward artificial neural network
with neurons fully inter-connected with neighbouring layers. The network is
reffered to as Multilayer Perceptron. The implementation supports computing
on both CPU and GPU using CUDA framework. The neural network as well as
training and validation process can be customized through command-line options.
For a complete list of available configuration options run `ffwdnet --help`.

The provided Makefile can be used to compile on Linux. For other platforms
you can create the build script yourself. Adjustments that are necessary in
the code should be minimal. To compile the following dependencies must be
installed in the system:

* **log4cpp** - logging framework for C+,+
* **gtest** - testing framework from Google Inc.,
* **CUDA SDK** - toolkit from NVIDIA for leveraging GPUs.
