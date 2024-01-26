import jax
from qtensor.Simulate import QtreeSimulator
from qtensor import QtreeQAOAComposer
import networkx as nx
import time
import numpy as np
import pickle


def qaoa_obj(G, p, sim):
    def fun(theta):
        beta = theta[:p]
        gamma = theta[p:]
        composer = QtreeQAOAComposer(G, gamma=gamma, beta=beta)
        composer.ansatz_state()
        circ = composer.circuit
        sim.simulate_batch(qc=circ, batch_vars=composer.n_qubits)

    return fun


graphs = pickle.load(open("../graphs.pickle", "rb"))
G = graphs[10]

sim = QtreeSimulator()

obj = qaoa_obj(G, 1, sim)
obj_jax = jax.jit(obj)
times = []
for i in range(5):
    theta = jax.numpy.array(np.random.rand(2*1))
    start = time.time()
    obj_jax(theta)
    stop = time.time()
    times.append(stop - start)
print(sum(times)/len(times))
print(times)
