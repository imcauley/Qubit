import quantum as q
import numpy as np

def superdense(a, b):
    input_state = q.tensor_product(q.comp_0, q.comp_0)

    circuit = []
    circuit.append(q.matrix_tensor(q.I, q.H))
    circuit.append(np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]))
    if(a):
        circuit.append(q.matrix_tensor(q.X, q.I))

    if(b):
        circuit.append(q.matrix_tensor(q.Z, q.I))

    circuit.append(np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]))
    circuit.append(q.matrix_tensor(q.I, q.H))

    print(q.circuit(input_state, circuit))


if __name__ == "__main__":
    superdense(0,0)