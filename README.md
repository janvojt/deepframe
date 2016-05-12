deepframe
=======

The `deepframe` application is purposed for building and training
deep neural networks. It supports multilayer perceptron network,
convolutional neural network, and deep belief network. Support for
additional network types can be added by implementing the required
layer types.

The implementation supports computing on both CPU and GPU using CUDA
framework. The neural network as well as training and validation process
can be customized through command-line options. For a complete list of
available configuration options run `deepframe --help`.

The provided Makefile can be used to compile the application on Linux.
For other platforms you can create the build script yourself.
Adjustments which are necessary in the code should be minimal. To
compile, the following dependencies must be installed in the system:

* **log4cpp** - logging framework for C++,
* **gtest** - testing framework from Google Inc.,
* **CUDA SDK** - toolkit from NVIDIA for leveraging GPUs.

Usage
=====

You can compile the application by running the `make` command in the
root directory. It will produce the executable binaries in the `bin`
folder. You can verify the application was built correctly by running
the unit tests, which actually require a working CUDA capable device.
The unit tests can be launched by the command below:

```
$ ./bin/deepframe-test
```

The application was built with "convention over configuration" approach
in mind. If you try to run the application without providing any
arguments, it will default to a small multilayer perceptron network,
which learns the XOR binary operator. You can train this simple
network by running the below command:

```
$ ./bin/deepframe
```

This will train the network on CPU. To use GPU simply add the `-p`
option.

To get familiar with the application and its configuration explore the
`examples` directory. It contains scripts created for solving some
sample problems. For example, to run a convolutional neural network,
which learns to recognize the handwritten digits from the MNIST dataset,
run the command below. This computation does take quite a bit of time
to complete, so you can again append the `-p` option to the end of the
command to make it faster by doing the heavy computations on GPU.

```
$ ./examples/mnist-conv.sh
```

The architecture of the network is defined via the network configuration
files. They allow you to build virtually any network architecture by
combining the supported layer types. To understand the syntax of the
configuration files consult the `examples` directory with sample
configurations of some common network architectures.
