# burgerPy
A simple mini-application that calls Python frameworks within a computational physics workflow

# Description

The purpose of this mini-application is to demonstrate how one may deploy scientific machine learning within a computational physics workflow. This code represents a *practical* deployment because it satisfies the following features:
1. The computation is performed using a compiled language as is the case with most legacy codes (C++).
2. We avoid disk-IO through in-situ transfer of data from the numerical computation to the machine learning computation (in Python).
3. Enable in-situ analysis on a GPU with zero-copy (i.e. avoid transfer to the host). 

In addition, this code also highlights the advantages of integrating the Python ecosystem with C++. We now have the following capabilities:
1. Utilizing arbitrary framework for accelerators such as CuPY through their Python APIs.
2. Easy in-situ visualization in matplotlib from a C++ computation.
3. A potential interface (if there are no issues with security) to streaming data from the internet (from say, a Python API).
4. Easy ability to save data using formats like HDF5 or NetCDF4.

The test-case demonstrated here aims to capture a modal decomposition using an SVD (Singular Value Decomposition). Here is how the coupling between C++ and Python is performed:
![Coupling](CouplingDiagram.png)

The test-case involves the solution of the 1-D Burger's equation. The problem is solved explicitly in time using the forward Euler method.  Computational kernels are written in CUDA to update the solution while CuPY is used perform in-situ analysis. 

## Requirements


## Building and Running


## Feature

### Using CuPY to enable zero-copy, in-situ analysis
In CuPY, `cupy.ndarray` is the counterpart of the NumPy `numpy.ndarray` which provides an interface for fixed-size multi-dimensional array which resides on a CUDA device.  Low-level CUDA support in CuPY allows us to retreive device memory. For example,

```
import cupy
from cupy.cuda import memory

  def my_function(a):
      b = cupy.ndarray(
                  a.__array_interface__['shape'][0],
                  cupy.dtype(a.dtype.name),
                  cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
                                             a.__array_interface__['data'][0], #<---Pointer?
                                             a.size,
                                             a,
                                             0), 0),
                  strides=a.__array_interface__['strides'])

``` 
