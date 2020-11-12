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

def matrix_tensor(A, B):
    # TODO: Generalize to any matrix
    A = A.flatten()
    B = B.flatten()
    output = np.zeros((4,4))

    output[0][0] = A[0] * B[0]
    output[0][1] = A[0] * B[1]
    output[0][2] = A[1] * B[0]
    output[0][3] = A[1] * B[1]

    output[1][0] = A[0] * B[2]
    output[1][1] = A[0] * B[3]
    output[1][2] = A[1] * B[2]
    output[1][3] = A[1] * B[3]

    output[2][0] = A[2] * B[0]
    output[2][1] = A[2] * B[1]
    output[2][2] = A[3] * B[0]
    output[2][3] = A[3] * B[1]

    output[3][0] = A[2] * B[2]
    output[3][1] = A[2] * B[3]
    output[3][2] = A[3] * B[2]
    output[3][3] = A[3] * B[3]

    return output


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

# Default direction is reverse
# So input the operators as a natural circuit diagram
def circuit(input_vector, circuit, reverse=True):
    if(reverse):
        direction = -1
    else:
        direction = 1
    output = input_vector

    for operator in circuit[::direction]:
        output = np.matmul(operator, output)

    return output

def simplify_circuit(circuit, reverse=True):
    output = np.identity(4)
    for operator in circuit[::-1]:
        output = np.matmul(operator, output)

    return output

def measure(a):
    pass

def omega(N):
    power = np.pi * (1j) * 2 / N
    return np.e ** power

def fourier(size, number):
    vector = np.zeros((size,), dtype=np.dtype('c16'))
    for i in range(size):
        current = np.zeros((size,), dtype=np.dtype('c16'))
        current[i] =  np.power(omega(size), ((number * i)))
        vector += current

    return np.sqrt(size) * vector

def fourier_transform(size):
    transform = np.zeros((size, size), dtype=np.dtype('c16'))
    for i in range(size):
        current_vector_a = np.transpose(fourier(size, i))
        current_vector_b = np.zeros((size,))
        current_vector_b[i] = 1
        # print(current_vector_b.shape)
        transform += np.outer(current_vector_a, current_vector_b)

    return 1 / np.sqrt(size) * transform

def translation_transform(size):
    transform = np.zeros((size, size))
    for i in range(size):
        current_vector_a = np.zeros((size,))
        current_vector_a[((i + 1) % size)] = 1
        current_vector_b = np.zeros((size,))
        current_vector_b[i] = 1
        transform += np.outer(current_vector_a, current_vector_b)

    return transform