import pyomo.environ as pyomo

model = pyomo.ConcreteModel()

model.x1 = pyomo.Var(domain=pyomo.NonNegativeReals)
model.x2 = pyomo.Var(domain=pyomo.NonNegativeReals)

model.c = pyomo.ConstraintList()
model.c.add(model.x1 * 10 + 1 >= model.x2)
model.c.add(model.x1 * 0.2 + 4 >= model.x2)
model.c.add(model.x1 * (-0.2) + 6 >= model.x2)

model.objective = pyomo.Objective(
    rule=lambda model: model.x1 + model.x2 * 10,
    sense=pyomo.maximize
)

# solver = pyomo.SolverFactory('gurobi')
solver = pyomo.SolverFactory('cbc')

result = solver.solve(model)

print(result)
print(model.x1(), model.x2())
