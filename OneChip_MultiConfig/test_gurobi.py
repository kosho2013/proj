from gurobipy import *


# opt_mod = Model(name='linear program')
# x = opt_mod.addVar(name='x', vtype=GRB.INTEGER, lb=0)
# y = opt_mod.addVar(name='y', vtype=GRB.INTEGER, lb=0)


# obj_fn = 5*x + 4*y
# opt_mod.setObjective(obj_fn, GRB.MINIMIZE)

# c1 = opt_mod.addConstr(x + y >= 8.8, name='c1')
# c1 = opt_mod.addConstr(2*x + y >= 10.22, name='c2')
# c1 = opt_mod.addConstr(x + 4*y >= 11.998, name='c3')


# opt_mod.optimize()
# opt_mod.write('linear_model.lp')


# print('objective value:', opt_mod.objVal)
# for v in opt_mod.getVars():
    # print(v.varName, v.x)
    
    
    
    
# w = [4,2,5,4,5,1,3,5]
# v = [10,5,18,12,15,1,2,8]
# N = len(w)
# C = 15

# model = Model('knapsack')

# x = model.addVars(N, vtype=GRB.BINARY, name='x')
# model.addConstr(sum(w[i]*x[i] for i in range(N)) <= C)
# obj = sum(v[i]*x[i] for i in range(N))
# model.setObjective(obj, GRB.MAXIMIZE)

# model.optimize()

# print('objective value:', model.objVal)
# for v in model.getVars():
    # print(v.varName, v.x)







# w = [4,2,5,4,5,1,3,5]
# v = [10,5,18,12,15,1,2,8]
# N = len(w)
# C = 15

model = Model('example')
d = model.addVar(vtype=GRB.INTEGER, name='d', lb=1)
part = model.addVar(vtype=GRB.INTEGER, name='part', lb=1)
V = model.addVar(vtype=GRB.INTEGER, name='V', lb=1)
ts = 1000
cap = 4000
magic = 3

alpha = model.addVar(vtype=GRB.INTEGER, name='alpha', lb=1)
model.addConstr(ts * d <= alpha * part * cap)
model.addConstr(alpha * part <= V)
model.addConstr(V <= 3)

model.setObjective(V, GRB.MINIMIZE)
model.optimize()

print('objective value:', model.objVal)
for v in model.getVars():
    print(v.varName, v.x)

















