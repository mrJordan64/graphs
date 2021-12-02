import numpy as np

d1 = np.array([[0, 1.1, 1.0, 1.4],
               [1.1, 0, 0.3, 1.3],
               [1.0, 0.3, 0, 1.2],
               [1.4, 1.3, 1.2, 0]])

d2 = np.array([[0, 1.9, 1.2, 0.4],
               [1.9, 0, 0.8, 2.1],
               [1.2, 0.8, 0, 1.2],
               [0.4, 2.1, 1.2, 0]])


def Qcriterion(d):
    n = len(d)
    Q = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            Q[i, j] = d[i, j] - (sum(d[i]) + sum(d[j]))/(n - 2)
            Q[j, i] = Q[i, j]
    return Q


def minOffDiagEntry(q):
    _min = q[0, 1]
    pos = [0, 1]
    for i in range(len(q)):
        for j in range(len(q)):
            if i != j and q[i, j] < _min:
                _min = q[i, j]
                pos = [i, j]

    return pos


def addLeaf(d):
    length = len(d)
    d = np.append(d, np.zeros((1, length)), axis=0)
    d = np.append(d, np.zeros((length + 1, 1)), axis=1)

    return d


def addNewDistances(d, leaves):
    for k in range(len(d)):
        d[-1, k] = (d[leaves[0], k] + d[leaves[1], k] - d[leaves[0], leaves[1]])/2
        d[k, -1] = d[-1, k]

    return d


def deleteLeaves(d, leaves):
    d = np.delete(d, leaves, 0)
    d = np.delete(d, leaves, 1)

    return d


d = d1
nodes = [i for i in range(len(d))]
nodeNumber = len(d)
tree = []

while len(d) > 3:
    q = Qcriterion(d)
    leaves = minOffDiagEntry(q)
    _sum = sum(d[leaves[0]]) - sum(d[leaves[1]])  # simplification. correct???
    # should be sum(d[leaves[0]]) - d[leaves[0], leaves[1]] - (sum(d[leaves[1]]) - d[leaves[0], leaves[1]])
    weight1 = d[leaves[0], leaves[1]] + _sum / (len(d) - 2)
    weight2 = d[leaves[0], leaves[1]] - _sum / (len(d) - 2)

    d = addLeaf(d)
    d = addNewDistances(d, leaves)
    d = deleteLeaves(d, leaves)

    tree.append([[nodes[leaves[0]], weight1],
                 [nodes[leaves[1]], weight2],
                 nodeNumber])

    nodes = np.delete(nodes, leaves)
    nodes = np.append(nodes, nodeNumber)
    nodeNumber += 1

else:
    weight1 = (d[0, 1] + d[0, 2] - d[1, 2]) / 2
    weight2 = (d[0, 1] + d[1, 2] - d[0, 2]) / 2
    weight3 = (d[0, 2] + d[1, 2] - d[0, 1]) / 2

    tree.append([[0:d0, 1:d1], d2])




