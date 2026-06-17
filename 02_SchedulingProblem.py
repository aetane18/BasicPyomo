import pyomo.environ as pyomo
import matplotlib.pyplot as plt

price_schedule = {
    0: 0.5,
    1: 0.6,
    2: 1.0,
    3: 1.0,
    4: 0.9,
    5: 1.1,
    6: 1.8,
    7: 1.5,
    8: 0.9,
    9: 0.8,
    10: 0.7,
    11: 1.0,
}

charge_schedule = {
    0: 0.,
    1: 0.,
    2: 0.,
    3: 0.,
    4: 0.3,
    5: 0.15,
    6: 0.15,
    7: 0.05,
    8: 0.05,
    9: 0.05,
    10: 0.,
    11: 0.,
}


plt.plot(range(12), [price_schedule[i] for i in range(12)], label="Market Price")
plt.plot(range(12), [charge_schedule[i] for i in range(12)], label="Charge Energy")
plt.xlabel("Time")
plt.ylabel("Normalized value")
plt.legend()
plt.show()


model = pyomo.ConcreteModel()

# ------------ Parameters and constant information
# number of time steps
model.nt = pyomo.Param(initialize=len(price_schedule), domain=pyomo.Integers)

# set of time steps
model.T = pyomo.Set(initialize=range(model.nt()))

# sales price at each time step
model.price = pyomo.Param(model.T, initialize=price_schedule)

# power added from charging at each time step
model.charge = pyomo.Param(model.T, initialize=charge_schedule)

# initial/maximum storage inventory
model.s0 = pyomo.Param(initialize=500.)

# maximum instantaneous power
model.wmax = pyomo.Param(initialize=150.)


# ------------ Variables
# power output
model.w = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)

# energy stored
model.s = pyomo.Var(model.T, domain=pyomo.NonNegativeReals)


# ------------ Objective function
def objective_func(model):
    # sum the price times power produced for all time steps
    return sum(model.w[t] * model.price[t] for t in model.T)

model.objective = pyomo.Objective(rule=objective_func, sense=pyomo.maximize)


# ------------ Constraints

# storage inventory is limited by capacity
def constr_store_capacity(model, t):
    return model.s[t] <= model.s0

model.constr_store_capacity = pyomo.Constraint(model.T, rule=constr_store_capacity)


# power output limited by maximum power
def constr_power(model, t):
    return model.w[t] <= model.wmax

model.constr_power = pyomo.Constraint(model.T, rule=constr_power)


# energy balance on storage based on power consumed
def constr_store_balance(model, t):
    if t == 0:
        # initial inventory of storage at time 0
        return model.s[t] == model.s0 - model.w[t] + model.charge[t] * model.s0
    else:
        return model.s[t] == model.s[t-1] - model.w[t] + model.charge[t] * model.s0

model.constr_store_balance = pyomo.Constraint(model.T, rule=constr_store_balance)


# set up the solver
solver = pyomo.SolverFactory('cbc')

# run the optimization
results = solver.solve(model, keepfiles=False, logfile="solve.log")


# Print and summarize the results
model.display()
print(f"time\tprice\tpower\tstorage")

for t in model.T:
    print(f"{t}\t{model.price[t]:.2f}\t{model.w[t]():>5.1f}\t{model.s[t]():>5.1f}")


# plot results
plt.bar(range(model.nt()), [model.w[t]() for t in model.T], label="power")
plt.ylabel("Power")
plt.xlabel("Time")
plt.legend()

ax = plt.gca().twinx()
ax.plot(range(model.nt()), [model.s[t]() / model.s0 for t in model.T], label="storage", color='red', marker='o')
ax.plot(range(model.nt()), [model.price[t] for t in model.T], label="price", color='gray', marker='o')
ax.set_ylabel("Price & Storage")

plt.legend()
plt.show()
