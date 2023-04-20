"""
Tensors: Learn about the matrix-like data structures
Tutorial page: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
"""

import torch
import numpy as np


"""
Initillizations: Learn about different ways to define a Tensor in PyTorch
"""

#=========================
# Initillizations:
# Native Python Lists -> PyTorch Tensors
#=========================
#Create example data
data = [
    [1, 2],
    [3, 4]
]
#Create a tensor object directly from the data
tensor_from_data = torch.tensor(data)
#Print the created Tensor
print(f"Tensor from data:\n{tensor_from_data}\n")

#=========================
# Initillizations:
# NumPy Arrays -> PyTorch Tensors
#=========================
#Create a numpy array
np_array = np.array(data)
#Create a Tensor object from NumPy array
tensor_from_npArr = torch.tensor(np_array)
#Print the created Tensor
print(f"Tensor from NumPy array:\n{tensor_from_npArr}\n")

#=========================
# Initillizations:
# PyTorch Tensors -> PyTorch Tensors
#=========================
#Copy Tensor from first example to use as example
tensor_example = tensor_from_data
#Create a Tensor that is like the example (shape and datatype), but override values to be all one
tensor_ones_override = torch.ones_like(tensor_example)
#Create a Tensor that is like the example (shape), but override values to be random and override the datatype to be floating-points
tensor_rand_override = torch.rand_like(tensor_example, dtype=torch.float)
#Print the created Tensors
print(f"Tensor from example Tensor:\n- Ones Override:\n{tensor_ones_override}\n- Rand Override:\n{tensor_rand_override}\n")

#=========================
# Initillizations:
# Defined Constants -> PyTorch Tensors
#=========================
#Define a constant for the shape of the Tensor (rows, columns)
const_shape = (2,3)
#Create a Tensor of random values with the defined shape
rand_tensor = torch.rand(const_shape)
#Create a Tensor of ones with the defined shape
ones_tensor = torch.ones(const_shape)
#Create a Tensor of zeros with the defined shape
zeros_tensor = torch.zeros(const_shape)
#Print the created Tensors
print(f"Tensor from defined constants:\n- Random Tensor with shape (2,3):\n{rand_tensor}\n- Ones Tensor with shape (2,3):\n{ones_tensor}\n- Zeros Tensor with shape (2,3):\n{zeros_tensor}\n")


"""
Attributes: Learn about the different data stored in a PyTorch Tensor object
"""

#=========================
# Attributes:
# Tensor Shape
#=========================
#Create a random tensor
tensor = torch.rand(3,4)
#Get the shape(dimensions) of the created tensor
tensor_shape = tensor.shape
#Print the shape of the tensor (expect it to be what we defined: (3,4))
print(f"Shape of Tensor: {tensor_shape}")

#=========================
# Attributes:
# Tensor Datatype
#=========================
#Create a random tensor
tensor = torch.rand(3,4)
#Get the datatype of the created tensor
tensor_datatype = tensor.dtype
#Print the datatype of the tensor (will be automatically defined, but can define it manually)
print(f"Datatype of Tensor: {tensor_datatype}")

#=========================
# Attributes:
# Tensor Device
#=========================
#Create a random tensor
tensor = torch.rand(3,4)
#Get the device that the created tensor is stored on (CPU or GPU via CUDA)
tensor_device = tensor.device
#Print the device that the tensor is stored on (by default it will be stored on the CPU, not taking advantage of parallel processing)
print(f"Device the Tensor is stored on: {tensor_device}\n")


"""
Operations: Learn about the operations in PyTorch that can be used on Tensor objects
"""

#=========================
# Operations:
# Change Tensor Device
#=========================
#Create a Tensor to work with
tensor = torch.rand(3,4)
#Print the current operation
print("Moving Tensor between devices:")
#Print the current device the Tensor is stored on
print(f"- Tensor is currently stored on: {tensor.device}")
#Indicate to user that change of Tensor device is in progress
print("- Now attempting to move Tensor to GPU...")
#Check to see if CUDA is available to use GPU
if(torch.cuda.is_available()):
    #If CUDA is available, move tensor to CUDA to utillize GPU
    tensor = tensor.to("cuda")
    #Print success of movement to user
    print("- Movement of Tensor onto GPU successful!")
#Define case for when CUDA is unavailable
else:
    #If CUDA unavailable, then indicate to the user that we failed to move the Tensor
    print("- Movement of Tensor onto GPU failed!")
#Now print the current device that the Tesnor is stored on to see if the Tensor movement worked or not
print(f"- Device of tensor is now stored on: {tensor.device}\n")

#=========================
# Operations:
# Indexing Tensors
#=========================
#Create a Tensor to work with
tensor = torch.ones(3,4)
#Print the current operation
print("Indexing Tensors:")
#Indexing of Tensors similar to NumPy indexing (example of printing the second column of a Tensor)
print(f"- Get 2nd column of the Tensor: {tensor[:, 1]}\n")

#=========================
# Operations:
# Concatinating Tensors
#=========================
#Create 2 Tensors to work with
tensor1 = torch.ones(2, 3)
tensor2 = torch.zeros(2, 3)
#Print the current operation
print("Concatinating Tensors:")
#Concatinate the 2 Tensors along along the 0th dimension (combine along same row (Top->Bottom))
tensor_combined_dim0 = torch.cat([tensor1, tensor2], dim=0)
#Concatinate the 2 Tensors along along the 1st dimension (combine along same column (Left->Right))
tensor_combined_dim1 = torch.cat([tensor1, tensor2], dim=1)
#Print the 2 cocatinated tensors
print(f"- Concatination along the 0th dimension:\n{tensor_combined_dim0}\n- Concatination along the 1st dimension:\n{tensor_combined_dim1}\n")

#=========================
# Operations:
# Transpose Tensors
#=========================
#Create a Tensor to work with
tensor = torch.rand(2, 3)
#Print the current operation
print("Transposing a Tensor:")
#Print the tensor
print(f"- Current Tensor:\n{tensor}")
#Transpose the tensor
tensor_transposed = tensor.T
#Print the transposed tensor
print(f"- Transposed Tensor:\n{tensor_transposed}\n")

#=========================
# Operations:
# Tensor Arithmetic (Scalar Addition)
#=========================
#Create a tensor to work with
tensor = torch.zeros(2, 3)
#Can perform element-wise addition via Scalar addition
#Method 1: operator overload
tensor_scalar_addition = tensor + 5
#Method 2: via member function
tensor_scalar_addition = tensor.add(5)

#=========================
# Operations:
# Tensor Arithmetic (Tensor Element-Wise Addition)
#=========================
#Create 2 Tensors to work with
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)
#Can perform Tensor Element-Wise Addition to add each element of similar sized Tensors
#Method 1: operator overload
tensor_tensor_addition = tensor1 + tensor2
#Method 2: member function
tensor_tensor_addition = tensor1.add(tensor2)

#=========================
# Operations:
# Tensor Arithmetic (Scalar Multiplication)
#=========================
#Create a tensor to work with
tensor = torch.rand(2, 3)
#Can perform element-wise multiplication via Scalar multiplication
#Method 1: operator overload
tensor_scalar_multiplication = tensor * 100
#Method 2: via member function
tensor_scalar_multiplication = tensor.mul(100)

#=========================
# Operations:
# Tensor Arithmetic (Tensor Element-Wise Multiplication)
#=========================
#Create 2 Tensors to work with
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)
#Can perform Tensor element-wise multiplication to multiply each element of similar sized Tensors
#Method 1: operator overload
tensor_tensor_addition = tensor1 * tensor2
#Method 2: member function
tensor_tensor_addition = tensor1.mul(tensor2)

#=========================
# Operations:
# Tensor Arithmetic (Tensor Matrix Multiplication)
#=========================
#Create 2 Tensors to work with
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)
#Can perform Tensor Matrix multiplication to multiply Tensors in a similar fashion to matrix multiplication (dim of col t1 needs to equal dim of row t2)
#Method 1: operator overload
tensor_tensor_addition = tensor1 @ tensor2.T
#Method 2: member function
tensor_tensor_addition = tensor1.matmul(tensor2.T)

#=========================
# Operations:
# In-Place Operations
#=========================
#Create 2 Tensors to work with
tensor1 = torch.rand(2, 3)
tensor2 = torch.rand(2, 3)
#In-Place operations have an underscore (_) as the suffix of the function name and directly change the Tensor value
#For each In-Place operation, there is a corrisponding "out-place" operation
#Directly add 5 to tensor1
tensor1.add_(5)
#Directly copy the data of tensor1 into tensor2
tensor2.copy_(tensor1)
#Directly transpose tensor2
tensor2.t_()


"""
Bridge with NumPy: Learn about how PyTorch Tensors and NumPy arrays are connected to one another
"""

#=========================
# Bridge with NumPy:
# PyTorch Tensor <-> NumPy Array
#=========================
#Creating a NumPy array from a PyTorch Tensor and vice versa share memeory locations (if the Tensor is on the CPU)
#This creates a biirectional connection where a change in one will see the same change on the other
#Create a PyTorch Tensor
tensor1 = torch.ones(5)
#Convert the Tensor to a NumPy array
npArr1 = tensor1.numpy()
#Create a NumPy array
npArr2 = np.ones(5)
#Convert the NumPy array to a Tensor
tensor2 = torch.from_numpy(npArr2)
#The .numpy() and the .from_numpy() commands are what creates a links and allows both objects to share the same memory location