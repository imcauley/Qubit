import quantum as q
import numpy as np

def kickback():
    circuit = []
    circuit.append(q.matrix_tensor(q.I, q.H))
    circuit.append(q.CNOT)
    circuit.append(q.matrix_tensor(q.I, q.H))

    input_vector = q.tensor_product(q.comp_1, q.comp_1)
    print(q.circuit(input_vector, circuit))
    # print(q.simplify_circuit(circuit))

if __name__ == "__main__":
    kickback()