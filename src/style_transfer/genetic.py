import random
from chromosome import Chromosome

def getPopulation(n):
    # get 100 chromosomes with n random genes
    population = [Chromosome.randomize(n) for _ in range(100)]
    # print each chrom in population
    print("Initial Population:")
    for i in range(100):
        print(" " + population[i].to_string())

    return population

def getMutation(child):
    # get a random point to mutate a gene
    mutation_point = random.randint(0, child.gene_count - 1)
    # flip the gene bit
    child.genes[mutation_point] ^= 1

    return child

def getChildren(parents):
    # get random dividing point for crossover
    crossover_cut = random.randint(1, parents[0].gene_count - 1)

    # get 2 sets of genes based on 2 different crossover cuts of the parents
    child_genes1 = parents[0].genes[:crossover_cut] + parents[1].genes[crossover_cut:]
    child_genes2 = parents[1].genes[:crossover_cut] + parents[0].genes[crossover_cut:]

    # create 2 child chroms with these gene sets
    child1 = Chromosome(child_genes1, parents[0].gene_count)
    child2 = Chromosome(child_genes2, parents[0].gene_count)

    # mutate both children
    child1 = getMutation(child1)
    child2 = getMutation(child2)

    # return set of children
    return [child1, child2]

def getParents(selection):
    # 2 highest fitnesses out of selection of 4
    highest_fit1 = -1
    highest_fit2 = -1
    # parents
    parent1 = None
    parent2 = None

    # go through each chrom in selection
    for chromosome in selection:
        fitness = chromosome.get_fitness()
        # gets the top 2 highest fitness chroms in selection
        if fitness > highest_fit1:
            highest_fit2 = highest_fit1
            parent2 = parent1
            highest_fit1 = fitness
            parent1 = chromosome
        elif fitness > highest_fit2:
            highest_fit2 = fitness
            parent2 = chromosome

    # 2 highest fitness become parents
    parents = []
    for chromosome in selection:
        if chromosome in [parent1, parent2]:
            parents.append(chromosome)

    return parents

def getSelection(population, gen , genes_num):
    new_population = []
    pop_indices = list(range(len(population)))

    # 25 selections
    for i in range(25):
        # 4 chroms per selection
        selection_indices = random.sample(pop_indices, 4)
        selection = [population[idx] for idx in selection_indices]

        # print each chrom in selection
        print("\nSelection #" + str(i + 1))
        for chromosome in selection:
            print(" " + chromosome.to_string())

        # get the 2 parents for this selection
        parents = getParents(selection)

        # get the children of the parents
        children = getChildren(parents)

        # selection is now parents and children
        selection = parents + children

        # print each chrom in selection
        print("\nSelection #" + str(i + 1) + " after breeding")
        for chromosome in selection:
            print(" " + chromosome.to_string())

        # add new selection to new population
        new_population += selection

    # shuffle new population
    new_pop_indices = list(range(len(new_population)))
    random.shuffle(new_pop_indices)
    new_population = [new_population[idx] for idx in new_pop_indices]

    for chrom in new_population:
        fitness = chrom.get_fitness()
        if fitness == genes_num:
            print("\nThe Royal Road has been found in generation " + str(gen) + ". Congratulations " + chrom.to_string())
            quit()

    return new_population


def main():
    # n = int(input("Please enter the number of genes you would like in each chromosome: "))
    genes_num = 500
    first_gen = getPopulation(genes_num)
    n = 2
    while True:
        print("\nGeneration: " + str(n))
        next_population = getSelection(first_gen, n, genes_num)
        first_gen = next_population
        n += 1


if __name__ == '__main__':
    main()

