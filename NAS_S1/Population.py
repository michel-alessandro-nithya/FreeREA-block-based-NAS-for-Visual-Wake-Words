from genotypes import *
from Exemplar import * 

def init_population(N, input_shape, max_flops=float('inf'), max_params=float('inf'), analyzer=None, start=0.0, metrics=None):
    population = []
    while len(population) < N:
        candidate = sample_exemplar(input_shape)
        if is_feasible(candidate, max_flops, max_params):
            population.append(candidate)
        if analyzer:
            analyzer.update(population, start, metrics)
    return population

def is_feasible(exemplar : Exemplar, max_flops=float('inf'), max_params=float('inf')):
    cost = exemplar.get_cost_info()
    return cost['MACs'] <= max_flops and cost['n_params'] <= max_params