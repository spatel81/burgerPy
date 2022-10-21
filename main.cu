#include "kernel.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define TPB 64

//Laplacian
__global__ void ddKernel(double *d_out, const double *d_in, int size, double h) {
  const int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i >= size - 1 || i == 0) return;
  d_out[i] = (d_in[i - 1] - 2.f*d_in[i] + d_in[i + 1]) / (h*h);
}

//First Derivative
__global__ void dKernel(double *d_out, const double *d_in, int size, double h) {
  const int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i >= size - 1 || i == 0) return;
  d_out[i] = (d_in[i + 1] - d_in[i - 1]) / (2*h);
}

//Burger's Update
//Could get better performance using shared mem
__global__ void burgerUpdate_Kernel(double *d_out, const double *d_in, int size, double c1, double c2) {

  const int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i > 0 && i < size-1 ) { 
     d_out[i] = d_in[i] + (d_in[i - 1] - 2.f*d_in[i] + d_in[i + 1])*c1 - d_in[i]*(d_in[i + 1] - d_in[i - 1])*c2;
  } else if (i == 0) {
     d_out[i] = d_in[size-2];
  } else if (i >= size-1) {
     d_out[i]=d_in[1];
  }  

}

constexpr int N = 256; // number of points in spatial discretization
void PyIt(PyObject *p_func, double *u);
void pynalyze(PyObject *p_func);

int main() {
  
  /****Some python initialization****/
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");
     std::cout << "Initialization of Python: Done" << std::endl;

  // initialize numpy array library
  import_array1(-1);

  PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
  PyObject* pModule = PyImport_Import(pName);
  Py_DECREF(pName); // finished with this string so release reference
     std::cout << "Loaded python module" << std::endl;

  std::cout << "Loading functions from module" << std::endl;
  PyObject* pcollect = PyObject_GetAttrString(pModule, "collection_func");
  PyObject* py_PlotField = PyObject_GetAttrString(pModule, "analyses_plotField");
  PyObject* py_SVD = PyObject_GetAttrString(pModule, "analyses_SVD");

  Py_DECREF(pModule); // finished with this module so release reference
     std::cout << "Loaded functions" << std::endl;

  /***********************************/

  const double PI = 3.1415926;	
  const int Ntests = 100;
  const double h = 2.0*PI/N;
  const double dt = 0.001; 
  const double FT = 2.000;  //Final Time
  const double NU = 0.01;   //diffusion param

  double s1 = dt*NU / (h*h);
  double s2 = dt / (2.0*h);

  double uh[N+2] = { 0.0 };
  double uh_prev[N+2] = { 0.0 };
  double result_parallel[N+2] = { 0.0 };

  double x;
  //Initialize
  for (int i = 1; i < N+1; ++i) {
    x    = 2.0*(i-1)*PI/N;
    uh[i]      = sin(x);
    uh_prev[i] = sin(x);
  }
  uh[0]   = uh[N]; // Ghost Nodes
  uh[N+1] = uh[1]; // Ghost Nodes
  
  uh_prev[0]   = uh_prev[N]; // Ghost Nodes
  uh_prev[N+1] = uh_prev[1]; // Ghost Nodes

  //Set-up some pointers and allocate device memory
  double *ud=0, *ud_prev=0;
  cudaMalloc(&ud, (N+2)*sizeof(double)); 
  cudaMalloc(&ud_prev, (N+2)*sizeof(double));
  cudaMemcpy(ud_prev, uh, (N+2)*sizeof(double), cudaMemcpyHostToDevice);

  double t = 0.0;
  auto walltime_start = std::chrono::high_resolution_clock::now();
  do{

      //Do the Burger's update with FD  
      burgerUpdate_Kernel<<<(N + TPB - 1)/TPB, TPB>>>(ud, ud_prev, N+2, s1, s2);
      {
        PyIt(pcollect, ud); //collect to global python data array  
      }
  
      //Move the current solution to the previous timestep 
      cudaMemcpy(ud_prev, ud, (N+2)*sizeof(double), cudaMemcpyDeviceToDevice);
      std::cout << "time = " << t << std::endl;
      t = t + dt;  

  }while(t<FT);
  auto walltime_finish = std::chrono::high_resolution_clock::now();
  double wallTime = std::chrono::duration<double,std::milli>(walltime_finish-walltime_start).count(); 
  std::cout << "avg. solver wallTime : " << wallTime/Ntests << std::endl;
 
  //copy result to host 
  cudaMemcpy(result_parallel, ud, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(ud);
  cudaFree(ud_prev);

      {
         Py_DECREF(pcollect);
      }

      //Plot the field
      {

        pynalyze(py_PlotField);  //collect to global python data array  
	Py_DECREF(py_PlotField);

      }

      //SVD
      {
	pynalyze(py_SVD); //Do Tensorflow stuff  
	Py_DECREF(py_SVD);
      }
}


//Python Wrappers
void PyIt(PyObject *p_func, double *u)
{
  PyObject* pArgs = PyTuple_New(1);

  //Numpy array dimensions
  npy_intp dim[] = {N+2};

  // create a new Python array that is a wrapper around u (not a copy) and put it in tuple pArgs
  PyObject* array_1d = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT64, u);
  PyTuple_SetItem(pArgs, 0, array_1d);

  // pass array into our Python function and cast result to PyArrayObject
  PyArrayObject* pValue = (PyArrayObject*) PyObject_CallObject(p_func, pArgs);
  //std::cout << "Called python data collection function successfully"<<std::endl;

  Py_DECREF(pArgs);
  Py_DECREF(pValue);
  // We don't need to decref array_1d because PyTuple_SetItem steals a reference
}


void pynalyze(PyObject *p_func)
{
  // panalsyses_func doesn't require an argument so pass nullptr
  PyArrayObject* pValue = (PyArrayObject*)PyObject_CallObject(p_func, nullptr);
  std::cout << "Called python analyses function successfully"<<std::endl;

  Py_DECREF(pValue);
  // We don't need to decref array_1d because PyTuple_SetItem steals a reference
}
