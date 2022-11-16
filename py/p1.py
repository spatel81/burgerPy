print("From python: Within python module")

import os,sys
HERE = os.getcwd()
sys.path.insert(0,HERE)

import numpy as np
import tensorflow as tf
import cupy
import matplotlib.pyplot as plt
from cupy.cuda import memory

print("PYTHON, Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_array = cupy.ndarray(shape=(2001,258))
x = np.arange(start=0,stop=2.0*np.pi,step=2.0*np.pi/256)
iternum = 0

def my_function1(a):
    b = cupy.asarray(a) #Host to device transfer
    b *= 5 
    b *= b 
    b += b 

def my_function2(a):
    b = cupy.ndarray(
                a.__array_interface__['shape'][0],
                cupy.dtype(a.dtype.name),
                cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
                                           a.__array_interface__['data'][0], #<---Pointer?
                                           a.size,
                                           a,
                                           0), 0),
                strides=a.__array_interface__['strides'])
    b *= 5 
    b *= b 
    b += b

#The collect function
def collect(a):
    global data_array,iternum
    b = cupy.ndarray(
                a.__array_interface__['shape'][0],
                cupy.dtype(a.dtype.name),
                cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
                                           a.__array_interface__['data'][0], #<---Pointer?
                                           a.size,
                                           a,
                                           0), 0),
                strides=a.__array_interface__['strides'])
    data_array[iternum,:]=b
    #print(data_array[iternum,:])
    iternum+=1
    return None

def analyze():
    global data_array,x
    dah=cupy.asnumpy(data_array) #Move from Device to Host to plot

    #Plot the Data
    plt.figure()
    for i in range(0,np.shape(dah)[0],400):
        plt.plot(x,dah[i,1:-1],label='Timestep '+str(i))
    plt.legend()
    plt.xlabel('x')
    plt.xlabel('u')
    plt.title('Field evolution')
    plt.savefig('Field_evolution.png')
    plt.close()

    #SVD on Device
    print('Performing SVD')
    u,s,v = cupy.linalg.svd(data_array[:,1:-1],full_matrices=False)

    # Plot SVD eigenvectors
    plt.figure()
    plt.plot(x, cupy.asnumpy(v[0,:]),label='Mode 0')
    plt.plot(x, cupy.asnumpy(v[1,:]),label='Mode 1')
    plt.plot(x, cupy.asnumpy(v[2,:]),label='Mode 2')
    plt.legend()
    plt.title('SVD Eigenvectors')
    plt.xlabel('x')
    plt.ylabel('v')
    plt.savefig('SVD_Eigenvectors_V.png')
    plt.close()

    return None

def analyze2():
    global data_array
    data_array_=data_array.toDlpack()
    print(data_array_)
    A=tf.experimental.dlpack.from_dlpack(data_array_)
    A.device
    x=A[0,:]
    y=A[0,:]
    #psum1 = x + y
    #psum2 = psum1 - x
    #psum3 = psum2 - y
    psum1 = tf.add(x,y)
    psum2 = tf.subtract(psum1,x)
    psum3 = tf.subtract(psum2,y)
    print(psum3)
    s=tf.reduce_sum(x)
    print(s)
    return None
