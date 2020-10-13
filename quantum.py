import numpy as np

# Computational Basis 
comp_0 = np.array([1,0])
comp_1 = np.array([0,1])

# Fourier Basis
four_pos = 1/np.sqrt(2) * np.array([1,1])
four_neg = 1/np.sqrt(2) * np.array([1,-1])

# Ciruclar Basis
circ_cw = 1/1/np.sqrt(2) * np.array([1, 1j])
circ_ccw = 1/1/np.sqrt(2) * np.array([1, -1j])

# Bell Basis
bell_00 = 1/np.sqrt(2) * np.array([1,0,0, 1])
bell_01 = 1/np.sqrt(2) * np.array([1,0,0,-1])
bell_10 = 1/np.sqrt(2) * np.array([0,1, 1,0])
bell_11 = 1/np.sqrt(2) * np.array([0,1,-1,0])

# Pauli Matrices
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])

# Common Operations
H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
CNOT = np.array([[1,0,0,0],[0,1,0,0], [0,0,0,1], [0,0,1,0]])

def outer_product(a, b):
    return np.outer(a,b)

def inner_product(a,b):
    return np.inner(a,b)
    
def tensor_product(a, b):
    c = np.zeros(a.shape[0] * b.shape[0])
    index = 0
    for x in a:
        for y in b:
            c[index] = x* np.conj(y)
            index+=1
    return c

def bra(*vectors):
    output = vectors[-1]
    for v in vectors[-2::-1]:
        output = tensor_product(v, output)

    return output

def ket(*vectors):
    output = vectors[-1]
    for v in vectors[-2::-1]:
        output = tensor_product(v, output)

    for (index, value) in output:
        output[index] = np.conj(value)

    output = np.transpose(output)

    return output


# a = np.zeros((16,16))
# a += outer_product(expand_comp(np.array([0,0,0,0])), expand_comp(np.array([0,0,0,0])))
# a += outer_product(expand_comp(np.array([0,0,0,1])), expand_comp(np.array([0,0,0,1])))
# a += outer_product(expand_comp(np.array([0,0,1,0])), expand_comp(np.array([0,0,1,0])))
# a += outer_product(expand_comp(np.array([0,0,1,1])), expand_comp(np.array([0,0,1,1])))

# a += outer_product(expand_comp(np.array([1,0,1,0])), expand_comp(np.array([1,0,0,0])))
# a += outer_product(expand_comp(np.array([1,0,0,0])), expand_comp(np.array([1,0,1,0])))

# a += outer_product(expand_comp(np.array([1,0,1,1])), expand_comp(np.array([1,0,0,1])))
# a += outer_product(expand_comp(np.array([1,0,0,1])), expand_comp(np.array([1,0,1,1])))

# a += outer_product(expand_comp(np.array([0,1,0,1])), expand_comp(np.array([0,1,0,0])))
# a += outer_product(expand_comp(np.array([0,1,0,0])), expand_comp(np.array([0,1,0,1])))

# a += outer_product(expand_comp(np.array([0,1,1,1])), expand_comp(np.array([0,1,1,0])))
# a += outer_product(expand_comp(np.array([0,1,1,0])), expand_comp(np.array([0,1,1,1])))

# a += outer_product(expand_comp(np.array([1,1,1,1])), expand_comp(np.array([1,1,0,0])))
# a += outer_product(expand_comp(np.array([1,1,1,0])), expand_comp(np.array([1,1,0,1])))
# a += outer_product(expand_comp(np.array([1,1,0,1])), expand_comp(np.array([1,1,1,0])))
# a += outer_product(expand_comp(np.array([1,1,0,0])), expand_comp(np.array([1,1,1,1])))

# print(np.matmul(a,expand_comp([0,1,1,0])))
# print(cnot(1/np.sqrt(2) * np.array([0,0,1,1])))
# print(tensor_product(comp_0, comp_1))
# print(expand_comp(np.array([0,0,0])))

a = tensor_product(four_pos, tensor_product(comp_0, comp_1))
b = tensor_product(comp_0, tensor_product(comp_1, four_pos))
c = tensor_product(comp_1, tensor_product(four_pos, comp_0))

d = tensor_product(four_neg, tensor_product(four_neg, four_neg))

# print(inner_product(d,d))
A = np.array([[1 ,1j],[1j, -1]])
AI = np.array([[1 ,-1j],[-1j, -1]])
print(np.matmul(A, AI))
