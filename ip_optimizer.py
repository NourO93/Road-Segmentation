from gurobipy import *
import numpy

'''
Given: 2D numpy array weight,
where weight[i,j] is an estimate of whether the 16x16 patch (i,j) is white (a road)
(it should be > 0 when we think it is white, and < 0 if not;
usually it would be obtained as log(p/(1-p)) where p is our probability estimate)
and parameter border_penalty.
Returns: 2D numpy array holding booleans: the optimal choice of white patches.
'''


def get_integer_programming_solution(weight, border_penalty):
    n, m = weight.shape
    total_cells = n * m  # maximum value of a flow f that is needed
    # vertices: (-1,-1) is the root (outside/border), others are (i,j)

    class Edge:
        # fields: a, b - endpoints,
        # fab, fba: flow variables (A to B, B to A),
        # z: discrepancy variable (|x_a - x_b|, only if a != root)
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def involves_root(self):
            return self.a == (-1, -1)

    edges = []
    for i in range(n):
        for j in range(m):
            if i+1 < n:
                edges.append(Edge((i,j), (i+1,j)))
            if j+1 < m:
                edges.append(Edge((i,j), (i,j+1)))
            if i == 0 or i == n-1 or j == 0 or j == m-1:
                edges.append(Edge((-1,-1), (i,j)))

    model = Model("model_name")
    model.setParam("LogToConsole", 0)
    model.setParam("LogFile", "tmp/gurobi.log")
    model.setParam("MIPGap", 0.005)  # 0.5% within optimum is enough
    model.setParam("TimeLimit", 20)  # 20 seconds is enough to wait...

    # create variables
    x = []
    for i in range(n):
        x.append([])
        for j in range(m):
            x[i].append(model.addVar(vtype=GRB.BINARY))  # x[i][j]
    for e in edges:
        e.fab = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        e.fba = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        if not e.involves_root():
            e.z = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

    model.update()

    # set objective
    obj = LinExpr()
    for i in range(n):
        for j in range(m):
            obj += weight[i, j] * x[i][j]
    for e in edges:
        if not e.involves_root():
            obj += - border_penalty * e.z
    model.setObjective(obj, GRB.MAXIMIZE)

    # create constraints

    # no flow into root
    for e in edges:
        if e.involves_root():
            model.addConstr(e.fba == 0)
    # flow only on x-active edges
    for e in edges:
        model.addConstr(e.fab <= total_cells * x[e.b[0]][e.b[1]])
        model.addConstr(e.fba <= total_cells * x[e.b[0]][e.b[1]])
        if not e.involves_root():
            model.addConstr(e.fab <= total_cells * x[e.a[0]][e.a[1]])
            model.addConstr(e.fba <= total_cells * x[e.a[0]][e.a[1]])
    # z_ab = abs(x_a - x_b)
    for e in edges:
        if not e.involves_root():
            model.addConstr(e.z >= x[e.a[0]][e.a[1]] - x[e.b[0]][e.b[1]])
            model.addConstr(e.z >= x[e.b[0]][e.b[1]] - x[e.a[0]][e.a[1]])
    # flow conservation
    for i in range(n):
        for j in range(m):
            inflow = LinExpr()
            outflow = LinExpr()
            for e in edges:
                if e.a == (i,j):
                    inflow += e.fba
                    outflow += e.fab
                if e.b == (i,j):
                    inflow += e.fab
                    outflow += e.fba
            model.addConstr(inflow - outflow >= x[i][j])

    # solve the problem
    model.optimize()

    result = numpy.zeros((n, m), dtype='bool')
    for i in range(n):
        for j in range(m):
            result[i,j] = x[i][j].x > 0.5  # should be close to 0 or 1

    return result


if __name__ == '__main__':
    # some simple test
    weight = numpy.array([[-4, -4, -4],
                                 [-4, 4.1, -4],
                                 [-4, -4, -4]])
    print(get_integer_programming_solution(weight, border_penalty=0.01))
