import random

class Chromosome:
    # track next chrom ID
    id_count = 0

    def __init__(self, genes, gene_count):
        self.genes = genes
        self.gene_count = gene_count
        # assigns unique ID to each chrom
        self.chrom_index = Chromosome.next_id()

    # generate a new Chromosome with random genes
    @classmethod
    def randomize(cls, gene_count):
        genes = [random.randint(0, 1) for _ in range(gene_count)]
        return cls(genes, gene_count)

    # increment id for each new chromosome
    @classmethod
    def next_id(cls):
        cls.id_count += 1
        return cls.id_count

    # calculate the fitness score (number of 1s in the genes)
    def get_fitness(self):
        return sum(self.genes)

    # format the chromosome genes and fitness score as a string
    def to_string(self):
        return "Chromosome: " + str(self.chrom_index) \
               + " Genes: " + ''.join(str(i) for i in self.genes) \
               + " Fitness: " + str(self.get_fitness())