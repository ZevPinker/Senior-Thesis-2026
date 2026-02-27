"""
Centralized Battery Coordination — Optimal Benchmark
======================================================
Solves the full joint LP for N batteries sharing a transformer,
given a 24-hour LMP price vector. This is the centralized optimum
against which ADMM will be benchmarked.

Assumptions:
  - No baseline generation (g = 0)
  - Initial SOC = 50% of E_max for all batteries
  - Terminal SOC = Initial SOC (state return constraint)
  - All batteries are identical (same parameters)
  - Single price node (one p_t vector for all batteries)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Parameters ────────────────────────────────────────────────────────────────

N = 5          # number of batteries
T = 24         # hours in scheduling horizon

E_max = 100.0      # kWh  — usable energy capacity
E_min = 0.0        # kWh
P_max = 50.0       # kW   — max charge / discharge rate (C-rate = 0.5)
eta_c = 0.95       # charge efficiency
eta_d = 0.95       # discharge efficiency
E_init = 0.5 * E_max   # 50% initial SOC
C = 80.0       # kW   — shared transformer capacity

# ── Price Vector ──────────────────────────────────────────────────────────────
# Replace this with your real LMP data.
# Shape: (T,) — one price per hour in $/MWh (or $/kWh, be consistent with E units).
# Example: synthetic price with morning and evening peaks.
np.random.seed(42)
p = np.array([
    30, 28, 27, 26, 25, 28,   # midnight–6am  (off-peak)
    38, 52, 68, 65, 55, 48,   # 6am–noon      (morning peak)
    45, 42, 40, 44, 58, 80,   # noon–6pm      (ramp + evening peak)
    90, 75, 60, 50, 42, 35,   # 6pm–midnight  (decay)
], dtype=float)

# ── Decision Variables ────────────────────────────────────────────────────────

c = cp.Variable((N, T), nonneg=True)   # charge rate    [kW], shape (N, T)
d = cp.Variable((N, T), nonneg=True)   # discharge rate [kW], shape (N, T)
E = cp.Variable((N, T + 1))            # state of charge [kWh], shape (N, T+1)
# E[:, 0]  = initial SOC
# E[:, T]  = terminal SOC

# ── Objective ─────────────────────────────────────────────────────────────────
# Revenue from discharging minus cost of charging, summed over all batteries
# and all hours. p is in $/MWh, power in kW → divide by 1000 for $ units,
# or keep in kWh-scale and interpret profit in $ directly (1 kWh at $/MWh = $/1000).
# For clarity we keep everything in kW / kWh and note units in comments.

# $ (if p in $/kWh) or $/1000 (if p in $/MWh)
profit = cp.sum(cp.multiply(p, d - c))
objective = cp.Maximize(profit)

# ── Constraints ───────────────────────────────────────────────────────────────

constraints = []

for i in range(N):

    # 1. SOC dynamics: E_{i,t+1} = E_{i,t} + eta_c * c_{i,t} - (1/eta_d) * d_{i,t}
    for t in range(T):
        constraints.append(
            E[i, t + 1] == E[i, t] + eta_c * c[i, t] - (1.0 / eta_d) * d[i, t]
        )

    # 2. SOC bounds
    constraints.append(E[i, :] >= E_min)
    constraints.append(E[i, :] <= E_max)

    # 3. Power bounds (non-negativity already handled by nonneg=True)
    constraints.append(c[i, :] <= P_max)
    constraints.append(d[i, :] <= P_max)

    # 4. Initial SOC
    constraints.append(E[i, 0] == E_init)

    # 5. State return constraint: terminal SOC = initial SOC
    constraints.append(E[i, T] == E_init)

# 6. Shared transformer constraint: sum of all power flows <= C at each hour
#    x_{i,t} = c_{i,t} + d_{i,t}  (total power through transformer)
for t in range(T):
    constraints.append(cp.sum(c[:, t] + d[:, t]) <= C)

# ── Solve ─────────────────────────────────────────────────────────────────────

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL, verbose=False)

if problem.status not in ["optimal", "optimal_inaccurate"]:
    raise RuntimeError(f"Solver failed with status: {problem.status}")

print(f"Status         : {problem.status}")
print(f"Total profit   : ${problem.value:,.2f}")
print(f"Profit per batt: ${problem.value / N:,.2f}")

c_val = c.value          # (N, T)
d_val = d.value          # (N, T)
E_val = E.value          # (N, T+1)
x_val = c_val + d_val    # total power per battery per hour

# Verify transformer constraint
total_power = x_val.sum(axis=0)   # (T,)
print(f"\nTransformer constraint check:")
print(f"  Max total power : {total_power.max():.2f} kW  (limit = {C} kW)")
print(
    f"  Constraint binding at {(total_power >= C - 0.01).sum()} of {T} hours")

# ── Plotting ──────────────────────────────────────────────────────────────────

BG = "#FAFAF8"
GRID = "#E5E3DC"
TEXT = "#2C2C2C"
SUBTEXT = "#7A7A72"
COLORS = ["#4A7FA5", "#C07A55", "#6B9E78", "#9B7EBD", "#C4A35A"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Georgia", "Times New Roman"],
    "axes.facecolor": BG, "figure.facecolor": BG, "axes.edgecolor": GRID,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
    "axes.labelcolor": TEXT, "axes.titlesize": 11, "axes.titlecolor": TEXT,
    "legend.frameon": True, "legend.framealpha": 0.92,
    "legend.edgecolor": GRID, "legend.fontsize": 8,
})

hours = np.arange(T)

# ── Figure 1: Price + Aggregate Power ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                               gridspec_kw={"hspace": 0.08})

ax1.plot(hours, p, color="#4A7FA5", lw=2)
ax1.set_ylabel("LMP ($/MWh)")
ax1.set_title(
    "Centralized Optimal Battery Dispatch — N={} Batteries".format(N))
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))

# Stacked area: charge (negative) and discharge (positive) per battery
charge_stack = np.zeros(T)
discharge_stack = np.zeros(T)
for i in range(N):
    ax2.fill_between(hours, discharge_stack, discharge_stack + d_val[i],
                     color=COLORS[i % len(COLORS)], alpha=0.75,
                     label=f"Battery {i+1}")
    discharge_stack += d_val[i]
    ax2.fill_between(hours, charge_stack, charge_stack - c_val[i],
                     color=COLORS[i % len(COLORS)], alpha=0.75)
    charge_stack -= c_val[i]

ax2.axhline(0, color=SUBTEXT, lw=0.8, linestyle="--", alpha=0.5)
ax2.axhline(C,  color="#B04040", lw=1.0, linestyle=":",
            alpha=0.7, label=f"Transformer limit ({C} kW)")
ax2.axhline(-C, color="#B04040", lw=1.0, linestyle=":", alpha=0.7)
ax2.set_ylabel("Power (kW)")
ax2.set_xlabel("Hour of Day")
ax2.set_xticks(hours)
ax2.legend(loc="upper left", ncol=3)
ax2.text(0.01, -0.14, "Positive = discharge (selling). Negative = charge (buying). Stacked by battery.",
         transform=ax2.transAxes, fontsize=7.5, color=SUBTEXT)

fig.tight_layout()
fig.savefig("centralized_dispatch.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
print("\nSaved → centralized_dispatch.png")
plt.close()

# ── Figure 2: SOC Trajectories ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))

for i in range(N):
    ax.plot(range(T + 1), E_val[i] / E_max * 100,
            lw=1.8, color=COLORS[i % len(COLORS)],
            marker="o", markersize=3, label=f"Battery {i+1}")

ax.axhline(50, color=SUBTEXT, lw=0.8, linestyle="--",
           alpha=0.6, label="Initial / Terminal SOC (50%)")
ax.axhline(100, color="#B04040", lw=0.8, linestyle=":", alpha=0.5)
ax.axhline(0,   color="#B04040", lw=0.8, linestyle=":", alpha=0.5)
ax.set_title("State of Charge Trajectories")
ax.set_ylabel("SOC (%)")
ax.set_xlabel("Hour")
ax.set_xticks(range(T + 1))
ax.set_ylim(-5, 110)
ax.legend(ncol=3)

fig.tight_layout()
fig.savefig("centralized_soc.png", dpi=200, bbox_inches="tight", facecolor=BG)
print("Saved → centralized_soc.png")
plt.close()

# ── Figure 3: Per-Battery Profit ──────────────────────────────────────────────
per_battery_profit = [(p * (d_val[i] - c_val[i])).sum() for i in range(N)]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(N), per_battery_profit,
              color=[COLORS[i % len(COLORS)] for i in range(N)],
              alpha=0.85, edgecolor=BG, linewidth=0.5)
ax.axhline(np.mean(per_battery_profit), color=SUBTEXT, lw=1.2,
           linestyle="--", label=f"Mean = ${np.mean(per_battery_profit):.2f}")
ax.set_title("Per-Battery Profit — Centralized Solution")
ax.set_ylabel("Profit ($)")
ax.set_xlabel("Battery")
ax.set_xticks(range(N))
ax.set_xticklabels([f"Battery {i+1}" for i in range(N)])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
ax.legend()

fig.tight_layout()
fig.savefig("outputs/centralized_profit.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
print("Saved → centralized_profit.png")
plt.close()
