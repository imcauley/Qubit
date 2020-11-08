import quantum as q
import numpy as np

def test():
    input_state = q.tensor_product(q.comp_0, q.four_pos)

    circuit = []
    circuit.append(q.matrix_tensor(q.H, q.I))
    circuit.append(q.CNOT)
    circuit.append(q.matrix_tensor(q.I, q.H))
    # circuit.append(np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]))
    # if(a):
    #     circuit.append(q.matrix_tensor(q.X, q.I))

    # if(b):
    #     circuit.append(q.matrix_tensor(q.Z, q.I))

    # circuit.append(np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]))
    # circuit.append(q.matrix_tensor(q.I, q.H))

    print(q.circuit(input_state, circuit))

if __name__ == "__main__":
    test()