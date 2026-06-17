import pyomo.environ as pyomo
import matplotlib.pyplot as plt

# ---------------- Data

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


# ---------------- Plot input data

plt.plot(range(12), [price_schedule[i] for i in range(12)], label="Market Price")
plt.plot(range(12), [charge_schedule[i] for i in range(12)], label="Charge Energy")
plt.xlabel("Time")
plt.ylabel("Normalized value")
plt.legend()
plt.show()


# ---------------- Model

model = pyomo.ConcreteModel()


# ---------------- Parameters and constant information

# number of time steps
model.nt = pyomo.Param(
    initialize=len(price_schedule),
    domain=pyomo.Integers
)

# set of time steps
model.T = pyomo.Set(
    initialize=range(model.nt())
)

# sales price at each time step
model.price = pyomo.Param(
    model.T,
    initialize=price_schedule
)

# power added from charging at each time step
model.charge = pyomo.Param(
    model.T,
    initialize=charge_schedule
)

# initial / maximum storage inventory
model.s0 = pyomo.Param(
    initialize=500.
)

# maximum instantaneous power
model.wmax = pyomo.Param(
    initialize=150.
)


# ---------------- Variables

# power output
model.w = pyomo.Var(
    model.T,
    domain=pyomo.NonNegativeReals
)

# energy stored
model.s = pyomo.Var(
    model.T,
    domain=pyomo.NonNegativeReals
)

# operational state: 1 if on, 0 otherwise
model.y = pyomo.Var(
    model.T,
    domain=pyomo.Binary
)


# ---------------- Objective function

def objective_func(model):
    # sum the price times power produced for all time steps
    return sum(model.w[t] * model.price[t] for t in model.T)

model.objective = pyomo.Objective(
    rule=objective_func,
    sense=pyomo.maximize
)


# ---------------- Constraints

# storage inventory is limited by capacity
def constr_store_capacity(model, t):
    return model.s[t] <= model.s0

model.constr_store_capacity = pyomo.Constraint(
    model.T,
    rule=constr_store_capacity
)


# energy balance on storage based on power consumed
def constr_store_balance(model, t):
    if t == 0:
        # initial inventory of storage at time 0
        return model.s[t] == model.s0 - model.w[t] + model.charge[t] * model.s0
    else:
        return model.s[t] == model.s[t - 1] - model.w[t] + model.charge[t] * model.s0

model.constr_store_balance = pyomo.Constraint(
    model.T,
    rule=constr_store_balance
)


# require the power cycle to run a minimum number of hours
def constr_min_runtime(model):
    return sum(model.y[t] for t in model.T) >= 8

model.constr_min_runtime = pyomo.Constraint(
    rule=constr_min_runtime
)


# if ON, power must be at least half of maximum power
def constr_run_mode_lower(model, t):
    return model.w[t] >= model.y[t] * model.wmax / 2

model.constr_run_mode_lower = pyomo.Constraint(
    model.T,
    rule=constr_run_mode_lower
)


# if OFF, power must be zero; if ON, power can go up to maximum power
def constr_run_mode_upper(model, t):
    return model.w[t] <= model.y[t] * model.wmax

model.constr_run_mode_upper = pyomo.Constraint(
    model.T,
    rule=constr_run_mode_upper
)


# ---------------- Solver

solver = pyomo.SolverFactory("cbc")

print("CBC available:", solver.available())

if not solver.available():
    raise Exception("CBC solver is not available. Install CBC or add it to PATH.")

# run the optimization
results = solver.solve(model, tee=True)


# ---------------- Print and summarize results

print("\nSolver Results")
print(results)

print("\nTermination condition:", results.solver.termination_condition)

print("\nObjective value:", pyomo.value(model.objective))

print(f"\ntime\tprice\tpower\tstorage\tstate")

for t in model.T:
    print(
        f"{t}\t"
        f"{pyomo.value(model.price[t]):.2f}\t"
        f"{pyomo.value(model.w[t]):>5.1f}\t"
        f"{pyomo.value(model.s[t]):>5.1f}\t"
        f"{int(pyomo.value(model.y[t])):d}"
    )


# ---------------- Plot results

plt.plot(
    range(model.nt()),
    [pyomo.value(model.w[t]) for t in model.T],
    label="power",
    marker="o"
)

plt.ylabel("Power")
plt.xlabel("Time")
plt.legend()

ax = plt.gca().twinx()

ax.plot(
    range(model.nt()),
    [pyomo.value(model.s[t]) / pyomo.value(model.s0) for t in model.T],
    label="storage",
    color="red",
    marker="o"
)

ax.plot(
    range(model.nt()),
    [pyomo.value(model.price[t]) for t in model.T],
    label="price",
    color="gray",
    marker="o"
)

ax.set_ylabel("Price & Storage")

plt.legend()
plt.show()