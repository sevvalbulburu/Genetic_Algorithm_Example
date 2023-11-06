import random
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


class Genetic():
    def __init__(self, population_number, gene_number, mutation_prob, file_name):
        self.population = []
        self.next_generation = []
        self.generation_count = 0       # iteration count
        self.population_number = population_number
        self.gene_number = gene_number  # represents N
        self.mutation_prob = mutation_prob  # probability
        self.found = False
        self.found_chromosome = []  # keeps the result
        self.words_info = dict()    # stores every unique word after filtering and their labels
        self.read_data(file_name)   # reads dataset
        self.gene_pool = list(self.words_info.keys())   # final key words
        # lists for creating plot
        self.iterations = []
        self.best_fitness = []
        self.avg_fitness = []


    def read_data(self, file_name):
        # open file and read lines as a list
        with open(file_name, "r", encoding = "utf-8") as file:
           content = file.readlines()

        # iterate lines
        for line in content:
            # finding label for sentence
            class_value = 0
            i = len(line)-1
            # iterate from end to start of line
            while i >= 0:
                # if char is a number it should be 0 or 1 which represents label
                if line[i].isnumeric():
                    if line[i] == '1':
                        class_value += 1
                    else:
                        class_value -= 1
                    i *= -1
                i -= 1

            # filtering alphabet chars, lowering every char
            filtered_line = ""
            line = line.replace("'s", "").lower()
            for char in line:
                if char == ' ' or char == '\'' or char.isalpha():
                    filtered_line += char

            # remove stopwords
            tokens = nltk.word_tokenize(filtered_line)
            words = [token for token in tokens if token not in stopwords.words('english')]

            # label counting: we increment if label is 1 decrement if label is 0
            # at the and if value is positive then it is thinked that the word is label 1 else label 0
            # if value is 0 we remove from dict out of loop 
            for word in words:
                word = word.strip('\'')
                if word in self.words_info:
                    self.words_info[word]["label_count"] += class_value
                    self.words_info[word]["count"] += 1
                else:
                    self.words_info[word] = dict()
                    self.words_info[word]["label_count"] = 0
                    self.words_info[word]["count"] = 0

        # if value is 0 we remove from dict out of loop 
        self.words_info = {k: v for k, v in self.words_info.items() if v["label_count"] != 0}

        # remove %10 of words which has the most openness between count and label count values
        tmp = [[k, v["count"]-v["label_count"]] for k, v in self.words_info.items()]
        tmp = sorted(tmp, key=lambda X:X[1])

        index = int(( 10 * len(tmp)) / 100)
        tmp2 = []
        tmp2.extend(tmp[:index])

        for word in tmp2:
            del self.words_info[word[0]]


    class Member:
        def __init__(self, chromosome):
            self.chromosome = chromosome
            self.fitness = 0


    def random_gene(self):
        # select a random word from pool
        Gene = random.choice(self.gene_pool)
        return Gene


    def create_chromosome(self):
        # creates a chromosome as list from random gene words
        chromosome = [self.random_gene() for i in range(self.gene_number)]
        return chromosome


    def calculate_fitness(self):
        for member in self.population:
            member.fitness = 0      # reset fitness value
            # first half
            for i in range(self.gene_number//2):
                # if positive then increment fitness value
                if self.words_info[member.chromosome[i]]["label_count"] > 0:
                    member.fitness += 1
                # if negative then decrement fitness value
                elif self.words_info[member.chromosome[i]]["label_count"] < 0:
                    member.fitness -= 1
            #second half
            for i in range(self.gene_number//2, self.gene_number):
                # if negative then increment fitness value
                if self.words_info[member.chromosome[i]]["label_count"] > 0:
                    member.fitness -= 1
                # if positive then decrement fitness value
                elif self.words_info[member.chromosome[i]]["label_count"] < 0:
                    member.fitness += 1

            # end condition
            try:
                if member.fitness == self.gene_number or \
                (self.generation_count > 700 and self.generation_count//2 < len(self.best_fitness)-self.best_fitness.index(max(map(lambda Member: Member.fitness, self.population)))):
                    self.found = True
                    self.found_chromosome = member.chromosome
            except ValueError:
                pass

    def crossover(self):
        # select best %80
        last_best = int(( 80 * self.population_number) / 100)

        # removing worst %20 population
        self.next_generation = []
        self.next_generation.extend(self.population[last_best:])

        # filling %20 with crossover result
        while len(self.next_generation) < self.population_number:
            # selecting 2 members to crossover
            member_1 = random.choice(self.population[last_best:]).chromosome
            member_2 = random.choice(self.population[last_best:]).chromosome
            new_member = []

            for gene1, gene2 in zip(member_1, member_2):
                # random.random() creates an value between 0-1
                if random.random() < self.mutation_prob:        # if probability is smaller than mutation rate mutate 
                    new_member.append(self.random_gene())
                elif random.random() < 0.5:                     # else chose random from one of them
                    new_member.append(gene1)
                else:
                    new_member.append(gene2)

            self.next_generation.append(self.Member(new_member))        # appending new member to generation

        self.population = self.next_generation      # population update


    def iteration_success(self):
        # store values to plot at every iteration
        self.iterations.append(self.generation_count)
        self.best_fitness.append(max(map(lambda Member: Member.fitness, self.population)))
        self.avg_fitness.append(sum(map(lambda Member: Member.fitness, self.population))/self.population_number)
        # print info
        print(f"Generation: {self.iterations[-1]}     best chromosome fitness: {self.best_fitness[-1]} \
    Success for best:{(self.best_fitness[-1]/self.gene_number)*100}     average chromosome fitness: {self.avg_fitness[-1]}")


    def graphic(self):
        # plot the data
        plt.plot(self.iterations, self.best_fitness, label='Best Fitness')
        plt.plot(self.iterations, self.avg_fitness, label='Average Fitness')

        # add axis labels and title
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Success Graph')

        # add a legend
        plt.legend()

        # show the plot
        plt.show()


    def main(self):
        # create a population first time randomly
        for i in range(self.population_number):
            self.population.append(self.Member(self.create_chromosome()))

        # iterate until target fitness achieved
        while not self.found:
            self.calculate_fitness()
            # to select best chromosomes from population we order population
            self.population = sorted(self.population, key=lambda Member: Member.fitness)
            self.iteration_success()
            self.crossover()
            self.generation_count += 1  # increment iteration count

        print()
        print(self.found_chromosome)
        self.graphic()


# preinstalls and defines
nltk.download('stopwords')
nltk.download('punkt')
datasets = ["geneticAlgorithmDatasets/amazon_cells_labelled.csv",
            "geneticAlgorithmDatasets/imdb_labelled.csv",
            "geneticAlgorithmDatasets/imdb_labelled2.csv",
            "geneticAlgorithmDatasets/yelp_labelled.txt",
            "geneticAlgorithmDatasets/yelp_labelled2.txt"]

# taking inputs1

index = int(input("""
Select dataset:
1-amazon_cells_labelled.csv
2-imdb_labelled.csv
3-imdb_labelled2.csv
4-yelp_labelled.txt
5-yelp_labelled2.txt
"""))-1

population_number = int(input("Population Number : "))
gene_number = int(input("Gene Number (N) : "))
mutation_prob = float(input("Mutation Probability Between 0-1 (Ex: 0.2) : "))

# calling functions
solve = Genetic(population_number, gene_number, mutation_prob, datasets[index])
solve.main()