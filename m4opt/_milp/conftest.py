collect_ignore_glob = []

try:
    import cplex  # noqa: F401
except ImportError:
    collect_ignore_glob.append("_cplex.py")

try:
    import gurobipy  # noqa: F401
except ImportError:
    collect_ignore_glob.append("_gurobi.py")
