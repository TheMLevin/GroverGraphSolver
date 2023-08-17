import json
import random

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, transpile
from qiskit_aer.noise import NoiseModel
from qiskit.test.mock import FakeVigo
from qiskit.circuit.library import MCMT
from qiskit.extensions import UnitaryGate
import numpy as np
import math
import functools
from typing import Dict, List, Tuple, Union, Collection, Any, Iterable
import itertools
import sympy as sp
from utils import *
import time
from qiskit_ionq import IonQProvider
from IonQAPIservice import IonQAPIservice

ionq_token = "tan6Mb4Qg7BOWwVPLC4XH6XKHjvOHroN"
provider = IonQProvider(ionq_token)
service = IonQAPIservice(ionq_token)

class Node:
    def __init__(self, name: Any, colors: Collection[Any]):
        self.name = name
        self.n = len(colors)
        self.j = math.ceil(math.log(self.n, 2))
        self.qubits = QuantumRegister(self.j, f'q_{name}')
        self.bits = ClassicalRegister(self.j, name)
        self.edges = []

        self.colors = set(sorted(colors))
        self.num_color = dict(enumerate(colors))
        self.color_num = {color: i for i, color in enumerate(colors)}
        self.initializer = self._make_init()
        count_ops = self.initializer.count_ops()
        self.cx_count = count_ops['cx'] if 'cx' in count_ops else 0

    def _make_init(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.qubits)
        i = 1
        while self.n % 2 ** i == 0:
            qc.ry(math.pi/2, i - 1)
            i += 1
        c = []
        for x in range(self.j - 1, i - 2, -1):
            if in_binary(self.n - 1, x):
                if self.j - 1 == x:
                    qc.ry(angle(self.n % 2 ** x, self.n), x)
                else:
                    qc.cry(angle(self.n % 2 ** x, self.n % 2 ** c[-1]), c[-1], x)
                c.append(x)
        qc.x(c[-1])
        for x in range(i - 1, self.j - 1):
            if x == c[-1]:
                qc.x(c[-1])
                del c[-1]
                qc.x(c[-1])
            qc.cry(math.pi/2, c[-1], x)
        if i - 1 != self.j:
            qc.x(-1)
        return qc


class Edge:
    _record = {}

    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b
        a.edges.append((b))
        b.edges.append((a))
        self.qubit = QuantumRegister(1, f"{a.name}-{b.name}")
        self.collisions = [(a.color_num[color], b.color_num[color]) for color in a.colors & b.colors]
        self.edge_oracle = self._make_oracle()

    def _reduce(self) -> List:
        a_bins = {x: f"{x:>0{self.a.j}b}" for x in self.a.num_color.keys()}
        b_bins = {x: f"{x:>0{self.b.j}b}" for x in self.b.num_color.keys()}
        X = [a_bins[a] + b_bins[b] for a in self.a.num_color.keys() for b in self.b.num_color.keys() if (a, b) not in self.collisions]
        Y = [a_bins[a] + b_bins[b] for a in self.a.num_color.keys() for b in self.b.num_color.keys()]
        W = []
        for k in range(1, self.a.j + self.b.j + 1):
            for J in itertools.combinations(range(self.a.j + self.b.j), k):
                for s in [f"{x:>0{k}b}"for x in range(2 ** k)]:
                    W.append((J, s))
            L = []
            for t in Y:
                L.append([int(''.join([t[i] for i in J]) == s) for (J, s) in W] + [int(t in X)])
            L = sp.Matrix(L)
            rref, pivots = L.rref()
            if len(W) not in pivots:
                result = [W[pivots[i]] for i, x in enumerate(rref[:, -1]) if x % 2]
                Edge._record[(tuple(self.a.colors), tuple(self.b.colors))] = result
                return result

    def _make_oracle(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.a.qubits, self.b.qubits, self.qubit)
        inverted = set()
        try:
            reduced = Edge._record[(tuple(self.a.colors), tuple(self.b.colors))]
        except (KeyError):
            reduced = self._reduce()
        for qubits, types in reduced:
            anti = {qubits[i] for i, x in enumerate(types) if x == '0'}
            qubits = set(qubits)
            cont = qubits - anti
            switch = (qubits & (anti - inverted)) | (inverted & cont)
            if switch:
                qc.x(switch)
            qc.compose(MCMT('x', len(qubits), 1), qubits | {-1}, inplace=True)
            inverted |= anti
            inverted -= cont
        if inverted:
            qc.x(inverted)
        return qc

class GraphSolver:
    def __init__(self, nodes: Collection[Node], edges: Collection[Edge], use_init: bool = True):
        self.nodes = nodes
        self.edges = edges
        self.ancilla = QuantumRegister(1, 'Ancilla')
        self.qubits = [x for y in [(node.qubits, node.bits) for node in nodes] for x in y] + [edge.qubit for edge in edges] + [self.ancilla]
        self.oracle = self._make_oracle()
        #self.oracle_cx = transpile(self.oracle, basis_gates=['u', 'cx']).count_ops()['cx']
        self.use_init = use_init
        self.initializer = self._make_init()
        self.diffuser = self._make_diffuser()
        self.m = 1
        self.lr = 1.2
        self.prev_j = set()
        self.max_m = int((math.pi / 4) * (prod(len(node.colors) for node in self.nodes) / min(len(node.colors) for node in self.nodes)) ** .5)

    def _estimate_j(self) -> int:
        while True:
            j = random.randint(1, self.m)
            if j not in self.prev_j or random.random() > 0.2:
                break
        self.prev_j.add(j)
        self.m = min(math.ceil(self.m * self.lr), self.max_m)
        return j

    def _make_edges(self) -> QuantumCircuit:
        qc = QuantumCircuit(*self.qubits)
        for edge in self.edges:
            qc.compose(edge.edge_oracle, [x for y in [edge.a.qubits, edge.b.qubits, edge.qubit] for x in y], inplace=True)
        return qc

    def _make_oracle(self) -> QuantumCircuit:
        qc = QuantumCircuit(*self.qubits)
        edge_oracle = self._make_edges()
        qc.compose(edge_oracle, inplace=True)
        qc.compose(MCMT('x', len(self.edges), 1), [edge.qubit[0] for edge in self.edges] + [self.ancilla[0]], inplace=True)
        qc.compose(edge_oracle, inplace=True)
        return qc

    def _make_init(self) -> QuantumCircuit:
        qc = QuantumCircuit(*self.qubits)
        if self.use_init:
            for node in self.nodes:
                qc.compose(node.initializer, node.qubits, inplace=True)
        else:
            for node in self.nodes:
                qc.h(node.qubits)
        return qc

    def _make_diffuser(self) -> QuantumCircuit:
        qc = QuantumCircuit(*self.qubits)
        qc.compose(self.initializer.inverse(), inplace=True)
        node_qubits = [x for y in [node.qubits for node in self.nodes] for x in y]
        qc.x(node_qubits)
        qc.compose(MCMT('x', sum([node.j for node in self.nodes]), 1), node_qubits + [self.ancilla[0]], inplace=True)
        qc.x(node_qubits)
        qc.compose(self.initializer, inplace=True)
        return qc

    def _solve(self, j) -> QuantumCircuit:
        qc = QuantumCircuit(*self.qubits)
        qc.compose(self.initializer, inplace=True)
        qc.h(self.ancilla)
        qc.z(self.ancilla)
        qc.barrier()
        for k in range(j):
            qc.compose(self.oracle, inplace=True)
            qc.barrier()
            qc.compose(self.diffuser, inplace=True)
            qc.barrier()
        for node in self.nodes:
            qc.measure(node.qubits, node.bits)
        return qc

    def check(self, colors: Dict[Node, Any]) -> bool:
        for edge in self.edges:
            if (colors[edge.a], colors[edge.b]) in edge.collisions:
                return False
            if colors[edge.a] not in edge.a.num_color:
                return False
            if colors[edge.b] not in edge.b.num_color:
                return False
        return True

    def run(self, simulate: bool = True, noise: bool = False) -> Tuple[Dict[Any, Any], QuantumCircuit, int, int]:
        count = 1
        now1 = time.time()
        while True:
            if simulate:
                backend = provider.get_backend('ionq_simulator')
                noise_model = 'aria-1' if noise else None
            else:
                raise Exception("Not Implemented")
            now = time.time()
            j = self._estimate_j()
            qc = self._solve(j)
            print(f"Solve {j}:", time.time() - now)
            now = time.time()
            jobid = service.submit_job(qc, backend=backend, noise_model=noise_model, shots=2)
            results = service.retrieve_job(jobid, wait_minutes=.5)['results']
            print(f"Execute:", time.time() - now)
            for result in results['counts'].keys():
                bits = len(result) // len(self.nodes)
                colors = {node: int(color[::-1], 2) for node, color in zip(self.nodes, (result.split(' ') if ' ' in result else (lambda gen: [itertools.islice(gen, node.j) for node in self.nodes])(x for x in result)))}
                if self.check(colors):
                    print("Done:", time.time() - now1)
                    return {color.name: color.num_color[num] for color, num in colors.items()}, qc, j, count
                count += j

    def run_dist(self, j: int, simulate: bool = True, noise: bool = False, shots: int = 100):
        if simulate:
            backend = provider.get_backend('ionq_simulator')
            noise_model = 'aria-1' if noise else None
        else:
            raise Exception("Not Implemented")
        qc = self._solve(j)
        jobid = service.submit_job(qc, backend=backend, noise_model=noise_model, shots=shots)
        results = service.retrieve_job(jobid, wait_minutes=1)['results']
        colorings = [({node: int(color, 2) for node, color in zip(self.nodes, key.split(' '))}, freq / shots) for key, freq
                     in results['counts'].items()]
        colors, freqs = list(zip(*sorted(colorings, key=lambda x: x[1])))
        corrects = [self.check(color) for color in colors]
        accuracy = 100 * sum(np.where(corrects, freqs, 0))
        labels = [[node.num_color[num] if num in node.num_color else 'X' for node, num in color.items()] for color in colors]
        plt.barh([str(label) for label in labels], freqs, color=['red' if correct else 'blue' for correct in corrects])
        plt.xlabel('Frequency')
        plt.ylabel('Coloring')
        plt.title(f"{accuracy:.2f}% correct")

    def reset(self):
        self.m = 1
        self.prev_j = set()


def run_experiment(nodes: Collection[Node], edges: Collection[Edge], K: int = 100, noise: bool = False) -> Tuple[List[int], List[int]]:
    counts = ([], [])
    graph1 = GraphSolver(nodes, edges, False)
    graph2 = GraphSolver(nodes, edges, True)
    for k in range(K):
        if not k % 1:
            print(" ", k)
        graph1.reset()
        graph2.reset()
        colors1, qc1, j1, count1 = graph1.run(noise=noise)
        counts[0].append(count1)
        colors2, qc2, j2, count2 = graph2.run(noise=noise)
        counts[1].append(count2)
    return counts


def run_experiments(node_range: Iterable[int], color_range: Iterable[int], K: int = 10, noise: bool = False):
    means = {}
    for color_num in color_range:
        print(color_num)
        colors = range(color_num)
        y1 = []
        y2 = []
        for node_num in node_range:
            print('', node_num)
            nodes = [Node(i, colors) for i in range(node_num)]
            edges = [Edge(nodes[i], nodes[i+1]) for i in range(node_num - 1)]
            count1, count2 = run_experiment(nodes, edges, K, noise)
            y1.append(sum(count1) / K)
            y2.append(sum(count2) / K)
            means[f"{color_num}-{node_num}"] = (count1, count2)
        plt.figure()
        plt.plot(node_range, y1, '-o', label='Original')
        plt.plot(node_range, y2, '-o', label='Modified')
        plt.legend()
        plt.title(f"Average Grover Repetitions for {color_num} Colors ({K} Simulations)")
        plt.xlabel('Nodes')
        plt.ylabel('Average Grover Reps')
        plt.ylim(bottom=0)
    data_log = f"{','.join(str(x) for x in color_range)}X{','.join(str(x) for x in node_range)}_data.txt"
    with open(data_log, 'w') as file:
        json.dump(means, file)