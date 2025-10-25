import random
import timeit

import torch


class Foo:
    def __init__(self, x):
        self.x = x

    def __call__(self, y):
        return self.x * y


class DAGNode:
    static_id = 0

    def __init__(self, neighbors: list['DAGNode'] = None):
        self.id = DAGNode.static_id
        DAGNode.static_id += 1
        self.neighbors: list['DAGNode'] = neighbors if neighbors is not None else []

    def __repr__(self):
        return f"({self.id})"

    def add(self, node: 'DAGNode'):
        self.neighbors.append(node)
        
    def remove(self, node: 'DAGNode'):
        try:
            self.neighbors.remove(node)
        except ValueError:
            pass


class DAG:
    def __init__(self, root: DAGNode):
        self.root = root if root is not None else DAGNode()

    def __repr__(self):
        def _repr(node: DAGNode, visited: set):
            if node in visited:
                return f"{node} -> [...]" if node.neighbors else f"{node}"
            visited.add(node)
            return f"{node} -> [{', '.join(_repr(neighbor, visited) for neighbor in node.neighbors)}]" if node.neighbors else f"{node}"

        return _repr(self.root, set())

    def sanitize(self):
        """Ensure that the DAG is acyclic and all nodes are reachable."""
        entered_nodes = set()
        left_nodes = set()
        self._dfs(self.root, self.root, entered_nodes, left_nodes)

    def _dfs(self, node: DAGNode, predecessor: DAGNode,
             entered_nodes: set, left_nodes: set):

        if node in entered_nodes and node not in left_nodes:
            predecessor.neighbors.remove(node)
            return
        if node in left_nodes:
            return

        entered_nodes.add(node)
        for neighbor in node.neighbors:
            self._dfs(neighbor, node, entered_nodes, left_nodes)
        left_nodes.add(node)


def test_dag():
    root = DAGNode()
    A = DAGNode()
    B = DAGNode([A])
    C = DAGNode([A, B])
    A.add(root)
    root.add(A)
    root.add(B)
    root.add(C)
    dag = DAG(root)
    dag.sanitize()
    print(dag)


def benchmark_dag(test_size=1000):
    root = DAGNode()
    nodes = [DAGNode() for _ in range(test_size)]
    for i in range(test_size):
        for j in range(i + 1, test_size):
            if random.randint(0,2) == 0:
                nodes[i].add(nodes[j])
    root.add(nodes[0])
    dag = DAG(root)
    duration = timeit.timeit(dag.sanitize, number=100)
    print(f"benchmark duration: {duration}")
    # print(dag)


def main():
    # foo = Foo(torch.tensor([2., 3.], requires_grad=True))
    # foo.dummy = 1
    # print (foo.dummy)
    benchmark_dag(100)
    benchmark_dag(1000)
    benchmark_dag(10000)

    a = torch.tensor([2.,3.], requires_grad=True)
    b = torch.tensor([6.,4.], requires_grad=True)
    r = 3*a**2
    r.backward(torch.tensor([1., 1.]))
    print(a.grad)


if __name__ == '__main__':
    main()
