# burgerPy
A simple mini-application that calls Python frameworks within a computational physics workflow

# Description

The purpose of this mini-application is to demonstrate how one may deploy scientific machine learning within a computational physics workflow. We claim that this code represents a *practical* deployment because it satisfies the following features:
1. The computation is performed using a compiled language as is the case with most legacy codes (C++).
2. We avoid disk-IO through in-situ transfer of data from the numerical computation to the machine learning computation (in Python).
3. Enable in-situ analysis on a GPU with zero-copy (i.e. avoid transfer to the host). 

In addition, this code also highlights the advantages of integrating the Python ecosystem with C++. We now have the following capabilities:
1. Utilizing arbitrary framework for accelerators such as CuPY through their Python APIs.
2. Easy in-situ visualization in matplotlib from a C++ computation.
3. A potential interface (if there are no issues with security) to streaming data from the internet (from say, a Python API).
4. Easy ability to save data using formats like HDF5 or NetCDF4.

The test-case demonstrated here aims to capture a modal decomposition using an SVD

