d1 = [[0.0, 1.1, 1.0, 1.4],
      [1.1, 0.0, 0.3, 1.3],
      [1.0, 0.3, 0.0, 1.2],
      [1.4, 1.3, 1.2, 0.0]]

d2 = [[0.0, 1.9, 1.2, 0.4],
      [1.9, 0.0, 0.8, 2.1],
      [1.2, 0.8, 0.0, 1.2],
      [0.4, 2.1, 1.2, 0.0]]

d3 = [[0.0, 0.06, 0.12, 0.14],
      [0.06, 0.0, 0.14, 0.12],
      [0.12, 0.14, 0.0, 0.22],
      [0.14, 0.12, 0.22, 0.0]]

d4 = [[0.000000, 0.014174, 0.040580, 0.059462, 0.064294, 0.291210, 0.067664, 0.219073],
      [0.014174, 0.000000, 0.046588, 0.064669, 0.070539, 0.298177, 0.074214, 0.226117],
      [0.040580, 0.046588, 0.000000, 0.078610, 0.091849, 0.338114, 0.096022, 0.246679],
      [0.059462, 0.064669, 0.078610, 0.000000, 0.102408, 0.365998, 0.106295, 0.260915],
      [0.064294, 0.070539, 0.091849, 0.102408, 0.000000, 0.257828, 0.030257, 0.233884],
      [0.291210, 0.298177, 0.338114, 0.365998, 0.257828, 0.000000, 0.252345, 0.445679],
      [0.067664, 0.074214, 0.096022, 0.106295, 0.030257, 0.252345, 0.000000, 0.238486],
      [0.219073, 0.226117, 0.246679, 0.260915, 0.233884, 0.445679, 0.238486, 0.000000]]


class DissimilarityMap:
    def __init__(self, matrix):
        self.matrix = matrix
        self.label = [i for i in range(len(matrix))]
        self.nextLabel = len(matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def reductionStep(self, leaves):
        n = len(self.matrix)
        self.matrix.append([0 for i in range(n + 1)])
        for i in range(n):
            newDist = (self[i][leaves[0]] + self[i][leaves[1]] - self[leaves[0]][leaves[1]]) / 2

            self[i].append(newDist)
            self[n][i] = newDist

        self.delete(leaves)

        self.label.append(self.nextLabel)
        self.nextLabel += 1

    def delete(self, leaves):
        for i in range(len(self[0])):
            del self[i][leaves[0]]
            del self[i][leaves[1]]

        del self.matrix[leaves[0]]
        del self.matrix[leaves[1]]

        del self.label[leaves[0]]
        del self.label[leaves[1]]


def Qcriterion(dissMap):
    n = len(dissMap)
    Q = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i, n):
            Q[i][j] = dissMap[i][j] - (sum(dissMap[i]) + sum(dissMap[j])) / (n - 2)
            Q[j][i] = Q[i][j]

    return Q


def minOffDiagEntry(q):
    _min = q[0][1]
    pos = [0, 1]
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            if q[i][j] < _min:
                _min = q[i][j]
                pos = [i, j]

    return pos


def treeNodeCherryReplace(tree, node, cherry):
    for i in range(len(tree)):
        if type(tree[i][0]) == list:
            treeNodeCherryReplace(tree[i][0], node, cherry)
        elif tree[i][0] == node:
            tree[i][0] = cherry


def _NJ_alg(d):
    n = len(d.matrix)
    decimals = 5
    if n == 3:
        w0 = round(d[0][1] + d[0][2] - d[1][2], decimals) / 2
        w1 = round(d[1][0] + d[1][2] - d[0][2], decimals) / 2
        w2 = round(d[2][0] + d[2][1] - d[0][1], decimals) / 2

        return [[d.label[0], w0], [d.label[1], w1], [d.label[2], w2]]

    elif n > 3:
        q = Qcriterion(d.matrix)
        leaves = minOffDiagEntry(q)
        leaves.sort(reverse=True)

        _sum = sum(d[leaves[0]]) - sum(d[leaves[1]])
        weight0 = round(d[leaves[0]][leaves[1]] + _sum / (n - 2), decimals) / 2
        weight1 = round(d[leaves[0]][leaves[1]] - _sum / (n - 2), decimals) / 2

        cherry = [[d.label[leaves[0]], weight0],
                  [d.label[leaves[1]], weight1]]

        newNode = d.nextLabel
        d.reductionStep(leaves)

        tree = _NJ_alg(d)
        treeNodeCherryReplace(tree, newNode, cherry)

        return tree


def NJ_alg(d):
    d = DissimilarityMap(d)
    return _NJ_alg(d)


print(NJ_alg(d4))
