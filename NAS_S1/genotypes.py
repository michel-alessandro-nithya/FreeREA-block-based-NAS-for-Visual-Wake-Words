import numpy as np

# Operations. In total, there are 3 different ConvNext, 9 different InvBottlneck, 3 different ResNet = 15 different operations
architecture_type = [ "ConvNext", "InvBottleneck", "ResNet"]
kernel_size = [3,5,7]
expand_ratio = [4,6,8]
layers_per_stage = [3,3,9,3]
stages = len(layers_per_stage)
## stride = [1,2]

def sample_operation():
    architecture_type_id = np.random.randint(0,len(architecture_type))
    kernel_size_id = np.random.randint(0,len(kernel_size))
    expand_ratio_id = np.random.randint(0,len(expand_ratio))
    ## stride_id = np.random.randint(0,len(stride))
    return [ 
        architecture_type[architecture_type_id], 
        kernel_size[kernel_size_id], 
        expand_ratio[expand_ratio_id], 
        ## stride[stride_id]
        ]


def sample_block(n : int ):
    # Sample one operation
    operation = sample_operation()
    # Build the block.
    return [ operation for _ in range(n) ]

def sample_network():
    # There are for stages, then 4 different block.
    # Since one block is defined by an operation, and there
    # are 15 total operations,
    # the search space amount to 15^4 possible different networks.
    return [ sample_block(n) for n in layers_per_stage ]

def sample_genotype():
    return[ sample_operation() + [n]  for n in layers_per_stage ]

def encode(network : list ) -> list:
    """Returns the genotype of the architecture"""
    return [network[stage][0] for stage in range(stages) ]

def mutate(genotype):
    # Stage containg the block to be mutated. Uniformly sampled
    block_choice = np.random.multinomial(1, np.array([1]*stages ) / stages )
    # Just get the one with outcome 1
    mutating_block_id = np.argmax(block_choice)

    # Mutate
    old_gene = genotype[mutating_block_id]
    different = False
    while ( not different ):
        # sample a gene
        new_gene = sample_operation() + [ layers_per_stage[mutating_block_id] ]
        # if it is equal to the actual gene, discard this mutation.
        different = not equal(new_gene,old_gene)

    new_genotype = [ *genotype ]
    new_genotype[mutating_block_id] = new_gene
    return new_genotype

def crossover(genotype1, genotype2):
    # Create copies of the genotypes to avoid modifying the original ones
    new_genotype1 = [*genotype1]
    new_genotype2 = [*genotype2]

    # Choose a random stage to perform crossover
    crossover_stage = np.random.randint(0, stages)

    # Perform crossover for the selected stage
    new_genotype1[crossover_stage] = genotype2[crossover_stage]
    new_genotype2[crossover_stage] = genotype1[crossover_stage]
    candidates = [new_genotype1, new_genotype2 ]

     # Randomly choiche one of the two genotypes
    return candidates [ np.random.binomial(1,0.5) ]


def equal(gene_1, gene_2) -> bool:
    if (gene_1[0] == gene_2[0]) : # same block
       if ( gene_1[0] != "InvBottleneck" ):
           return gene_1[1] == gene_2[1] # expand_ratio is not significant for other blocks
       else :
           return (gene_1[1] == gene_2[1]) and (gene_1[2] == gene_2[2])
    else : 
        return False 