import networkx as nx
import pulp

nodes = (
    (1, 5),
    (2, 6),
    (3, 9),
    (4, 12),
    (5, 7),
    (6, 12),
    (7, 10),
    (8, 6),
    (9, 10),
    (10, 9),
    (11, 7),
    (12, 8),
    (13, 7),
    (14, 5),
)

edges = [
    (1, 2),
    (2, 4),
    (4, 7),
    (7, 10),
    (10, 12),
    (12, 14),
    (1, 3),
    (3, 6),
    (3, 5),
    (6, 9),
    (6, 8),
    (5, 9),
    (5, 8),
    (9, 11),
    (8, 11),
    (11, 12),
    (11, 13),
    (13, 14),
]

cpm = nx.DiGraph()

for n, p in nodes:
    cpm.add_node(n, p=p)

cpm.add_edges_from(edges)

cpm.add_node("Cmax", p=0)
cpm.add_edges_from([(j, "Cmax") for j in cpm.nodes()])

prob = pulp.LpProblem("Critical_Path_Problem", pulp.LpMinimize)

all_nodes = [j for j in cpm.nodes()]
s = pulp.LpVariable.dicts("s", all_nodes, 0)  # start
c = pulp.LpVariable.dicts("c", all_nodes, 0)  # completion

for j in cpm.nodes():
    prob += c[j] >= s[j] + cpm.nodes[j]['p']

for j in cpm.nodes():
    for i in cpm.predecessors(j):
        prob += s[j] >= s[i] + cpm.nodes[i]['p']

for j in cpm.nodes():
    for i in cpm.predecessors(j):
        prob += c[j]  >= c[i] + cpm.nodes[j]['p']

eps = 1e-5
prob += (
    c["Cmax"]
    + eps * pulp.lpSum([s[j] for j in cpm.nodes()])
    - eps * pulp.lpSum([c[j] for j in cpm.nodes()])
)

# prob.writeLP("cpmLP.lp")
prob.solve()
pulp.LpStatus[prob.status]

for j in cpm.nodes():
    print(
        j, s[j].varValue, c[j].varValue, c[j].varValue - s[j].varValue - cpm.nodes[j]['p']
    )
