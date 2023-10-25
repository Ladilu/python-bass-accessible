# The following code shows that the degree sequence generated
# with the command “nx.expected_degree_graph” does not closely follow a power law

import networkx as nx
import matplotlib.pyplot as plt

N=1000 # number of nodes

# first we generate a sequence of N random values distributed according to a power law
# in this example the power law exponent is 3
# the sequence is regarded as a degree sequence

s = nx.utils.powerlaw_sequence(N, 3)

# then we generate a graph with those expected degrees with the Chung-Lu method
# note that the values of the degrees do not need to be integer

G1 = nx.expected_degree_graph(s, selfloops=False)

# we extract from the graph its real degree sequence

deg_sequence=sorted((d for n, d in G1.degree()), reverse=True)
print(deg_sequence)

# we count the number of nodes with a certain degree, in this example degree 5
# and those with double degree

count5 = deg_sequence.count(5)
count10 = deg_sequence.count(10)
print(count5)
print(count10)
print(count5/count10)

# if the power law were respected, the ratio count5/count10 should be on average equal to 2^3=8
# but this is not verified
