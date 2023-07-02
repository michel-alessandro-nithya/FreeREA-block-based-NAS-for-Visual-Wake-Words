from Exemplar import Exemplar
from Population import *
from metrics import *
import numpy as np 


METRIC_NAME_MAP = {
    # log(x)
    'logsynflow': compute_synflow_per_weight,
    # x
    'synflow': lambda n, inputs, targets, dev: compute_synflow_per_weight(n, inputs, targets, dev, remap=None),

    'params': lambda n, _1, _2, _3: count_params(n) / 1e6,
    'macs': lambda n, inp, _1, _2: get_macs_and_params(n, inp.shape)[0],
    'naswot': compute_naswot_score,
}

def prune(population : list,
          history : list ,
          max_flops : float,
          max_params : float ,
          metrics : list,
          random_batch,
          feasibility = True,
          K : int = 0,
        ):

    # Whether real world constraints have to be taken into account.
    if feasibility:
        population, history = prune_flops_params(population, history, max_flops, max_params)
    
    # REA
    population = kill_oldest(population, K)
    population, history = prune_tfm(population, history, K, metrics, random_batch)
    return population, history


def update_population_history(population : list, history : list, replace = True):
    
    if replace:
        # Update already seen genotypes
        history.update({exemplar.get_genotype_str(): exemplar for exemplar in population})
    else:
        # Add new genotypes
        history.update({exemplar.get_genotype_str(): exemplar for exemplar in population if exemplar.get_genotype_str() not in history})
        # For already seen exemplars, just update the generation
        for exemplar in population:
            history[exemplar.get_genotype_str()].set_generation(exemplar.get_generation())

    # This is also removing exemplars with same genotype from population
    current_genotypes = set([exemplar.get_genotype_str() for exemplar in population])
    population = [history[genotype] for genotype in current_genotypes]
    return population, history


def clean_history(history : list, max_flops : float, max_params : float):
    return {genotype: exemplar for genotype, exemplar in history.items()
            if is_feasible(exemplar, max_flops = max_flops, max_params = max_params) }

def prune_flops_params(population, history, max_flops, max_params):
    population, history = update_population_history(population,history)
    history = clean_history(history, max_params, max_flops)
    return [ exemplar for exemplar in population if is_feasible(exemplar,max_flops,max_params) ], history

def kill_oldest(population, K, R = 1):

    # Get exemplars'generations
    generations = [exemplar.get_generation() for exemplar in population]

    # first iteration (all eexemplars belong to generation 0), do not perform ageing 
    if ( not np.any( generations ) ):
        return population
    
    # kill the oldest
    for _ in range(R):
        if len(population) > K + R - 1: # K = N = 25
            oldest = np.argmin(generations)
            population.pop(oldest)
            generations = [exemplar.generation for exemplar in population]

    return population

def prune_tfm(population=None, history=None, K=5, metrics = [], random_batch = []):
    # training free metrics

    # Now, exemplars are going to be evaluated. Thus, if it is the first iteration, they born.
    for exemplar in population:
        exemplar.born = True

    population, history = update_population_history(population, history)

    if K >= len(population):
        # No further pruning needed.
        return population, history

    return return_top_k(population, K, metrics, random_batch), history


    
def return_top_k(exemplars, K, metric_names, batch_input) :
    exemplars = [exemplar for exemplar in exemplars if exemplar.born]
     
    values_dict = {}

    # compute metrics
    for metric in metric_names : 
        values_dict[metric] = np.array( [exemplar.get_eval_metrics(batch_input)[metric] for exemplar in exemplars ])
    
    # initialize to zero
    scores = np.zeros(len(exemplars))

    scores_dict = {}

    # get score
    for metric_n in metric_names:
        scores_dict[metric_n] = values_dict[metric_n] / (np.max(np.abs(values_dict[metric_n])) + 1e-9)
        scores += scores_dict[metric_n]

    # set rank 
    for _, (exemplar, rank) in enumerate(zip(exemplars, scores)):
        exemplar.rank = rank
    
    # The higher the better
    exemplars.sort(key=lambda x: -x.rank)

    return exemplars[:K]



