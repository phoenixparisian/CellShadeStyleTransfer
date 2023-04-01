import random
import numpy as np
from chromosome import Chromosome
import matplotlib.pyplot as plt
from PIL import Image as img
MAX_GENERATION = 5000
POPULATION_SIZE = 100
def getPopulation(n):
    # get 100 chromosomes with n random genes
    population = [Chromosome(Chromosome.generate()) for _ in range(POPULATION_SIZE)]
    # print each chrom in population
    # print("Initial Population:")
    # for i in range(10): #POPULATION SIZE
    #     print(" " + population[i].to_string())


    return population

def getMutation(child):
    # get a random point to mutate a gene
    mutation_pixel = random.randint(0, child.gene_count - 1)
    mutation_point = random.randint(0, len(child.genes[mutation_pixel])-1)
    
    # flip the gene bit
    # print( "CHILDREN GENES", child.genes , "CHILDREN GENES")
    child.genes[mutation_pixel][mutation_point]= random.random()

    return child

def getChildren(parents):
    # get random dividing point for crossover
    crossover_cut = random.randint(1, parents[0].gene_count - 1)

    # get 2 sets of genes based on 2 different crossover cuts of the parents
    child_genes1 = parents[0].genes[:crossover_cut] + parents[1].genes[crossover_cut:]
    child_genes2 = parents[1].genes[:crossover_cut] + parents[0].genes[crossover_cut:]

    # create 2 child chroms with these gene sets
    # child1 = Chromosome(child_genes1, parents[0].gene_count)
    # child2 = Chromosome(child_genes2, parents[0].gene_count)
    child1 = Chromosome(child_genes1)
    child2 = Chromosome(child_genes2)

    # mutate both children
    child1 = getMutation(child1)
    child2 = getMutation(child2)

    # return set of children
    return [child1, child2]

def getParents(selection):
    # 2 highest fitnesses out of selection of 4
    highest_fit1 = 100000000.00
    highest_fit2 = 100000000.00
    # parents
    parent1 = None
    parent2 = None

    
    # go through each chrom in selection
    for chromosome in selection:
        fitness = chromosome.get_fitness()
        # gets the top 2 highest fitness chroms in selection
        if fitness < highest_fit1:
            highest_fit2 = highest_fit1
            parent2 = parent1
            highest_fit1 = fitness
            parent1 = chromosome
        elif fitness < highest_fit2:
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
    random.shuffle(pop_indices)

    # 25 selections
    for i in range(25):
        # 4 chroms per selection
        selection_indices = random.sample(pop_indices, 4)
        selection = [population[idx] for idx in selection_indices]

        # print each chrom in selection
        # print("\nSelection #" + str(i + 1))
        # for chromosome in selection:
        #     print(" " + chromosome.to_string())

        # get the 2 parents for this selection
        
        parents = getParents(selection)
        
        # get the children of the parents
        children = getChildren(parents)

        # selection is now parents and children
        selection = parents + children

        # print each chrom in selection
        # print("\nSelection #" + str(i + 1) + " after breeding")
        # for chromosome in selection:
        #     print(" " + chromosome.to_string())

        # add new selection to new population
        new_population += selection

    # shuffle new population
    new_pop_indices = list(range(len(new_population)))
    random.shuffle(new_pop_indices)
    new_population = [new_population[idx] for idx in new_pop_indices]

    # for chrom in new_population:
    #     fitness = chrom.get_fitness()
    #     if fitness == genes_num:
    #         # print("\nThe Royal Road has been found in generation " + str(gen) + ". Congratulations " + chrom.to_string())
    #         quit()

    return new_population


def main():
    # n = int(input("Please enter the number of genes you would like in each chromosome: "))
    genes_num = 10
    first_gen = getPopulation(genes_num)
    n = 2
    while True:
        print("\nGeneration: " + str(n))
        print(genes_num)
        next_population = getSelection(first_gen, n, genes_num)
        first_gen = next_population
        n += 1
        print("GENERATION" , n)
        if(n == MAX_GENERATION): 
            fittest = 1000000000
            index = 0
            for i in range(len(first_gen)): 
                if(fittest>first_gen[i].get_fitness()): 
                    index = i
                    fittest = first_gen[i].get_fitness()
            # first_gen[index].to_string()
            display = []
            for ii in range(len(first_gen[index].genes)):
                if(type(first_gen[index].genes[ii])!=float):
                    display.append(first_gen[i].genes[ii])
            plt.imshow(display)
            plt.show()

            break
                



if __name__ == '__main__':
    main()

