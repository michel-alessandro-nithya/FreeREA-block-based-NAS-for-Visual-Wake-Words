from Population import *
from prune import *
import random
import time

def search(n_iter,N=25, batch_size = [64], input_shape = [3,128,128], n=5, P=[1], R=[1], max_flops=200, max_params=2.5, max_time=2, n_random=0, analyzer=None):
    """
    :param api:             Search space API
    :param N:               Size of initial random population
    :param K:               Size of surviving population at each step
    :param P:               Parallel mutations
    :param R:               Consecutive mutations
    :param max_flops:       Maximum flops (1e6)
    :param max_params:      Maximum parameters 
    :param steps:           Number number of steps
    :return:
    """

    # start = time.time()


    metrics = ["NASWOT","LogSynFlow"]

    random_batch = torch.rand(batch_size + input_shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_batch = random_batch.to(device)
    print(random_batch.device)

    # Initialize population of N feasible exemplars
    print("Generating population...")
    population = init_population(N, [1] + input_shape, max_flops, max_params, analyzer= None)

    # Initialize history
    print("Initializing history...")
    history = { exemplar.get_genotype_str() : exemplar for exemplar in population }

    # Prune according to feasibility, ageing and training free metrics
    print("First Pruning...")
    population, history = prune(population, history, max_flops=max_flops,
                               max_params=max_params, K=N, metrics=metrics, random_batch = random_batch)

    step = 0
    for i in range(n_iter):
        print("**************************************************")
        print(f"\t Iteration number {i + 1} ")

        suitable = False
        j = 0 
        # Keep generating exemplars until at least one new feasible exemplars is generated
        while not suitable:
            j += 1 
            print(f" Attempnt to generate : {j}")
            ## Improved tournament selection

            # Sample n (5) elements from our population
            sampled = random.sample(population, n)

            # Return the two best exemplar among the n selected
            top_1,top_2 = return_top_k(sampled, 2, metrics, random_batch)
            print(f"1st Best architecture elected = {top_1.get_genotype_str()}")
            print(f"1st Best architecture score = {top_1.rank}")
            print(f"2nd Best architecture elected = {top_2.get_genotype_str()}")
            print(f"2nd Best architecture score = {top_2.rank}")

            # Mutate both of them
            population += [ top_1.mutate(generation = step + 1) ]
            population += [ top_2.mutate(generation = step + 1) ]

            # Crossover
            population += [ crossover(top_1,top_2, generation = step + 1 ) ]

            # Update history
            population, history = update_population_history(population,history,replace=False)
            
            # Feasibility prune. Note: only the three new child can be pruned.
            population, history = prune_flops_params(population,history, max_flops, max_params)

            # if the population is again N, no feasible exemplar have been generated
            suitable = len(population) > N
        
        # Up to this moment, history may contains some not born exemplar
        # since an exemplar is considered bord only if feasible

        # ageing and training free metrics pruning
        population, history = prune(population, history, max_flops=max_flops, max_params=max_params,
                                   K=N, metrics=metrics, feasibility=False, random_batch= random_batch)
        
        step += 1
    return population,history



