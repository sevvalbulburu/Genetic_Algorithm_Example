# Genetic_Algorithm_Example

## Project Explanation
[Pdf explanation](explanation.pdf)

## Install dependencies
-	Install nltk and matplotlib libraries by typing
```
		$ pip3 install nltk
```
```
		$ pip3 install matplotlib
```

## Code Explanation

This code implements a genetic algorithm to solve a binary classification problem for a given dataset. The goal is to find a set of N keywords (genes) that can accurately predict the label of each sentence in the dataset. The code first reads the dataset, cleans and preprocesses the text, and then generates a pool of unique keywords (words_info) that are not stopwords and have a significant difference between the number of times they appear in the positive and negative classes. The genetic algorithm then generates a population of chromosomes (lists of N genes) where each gene is randomly selected from the pool. The algorithm then calculates the fitness of each chromosome based on how well it predicts the labels of the sentences in the dataset. The fitness function rewards chromosomes whose first half of genes are more prevalent in positive sentences, and the second half of genes are more prevalent in negative sentences. The algorithm then selects the best chromosomes, performs crossover between pairs of them, and mutates the offspring with a certain probability. The process of selection, crossover, and mutation continues until the algorithm finds a chromosome with a fitness score equal to N or the maximum number of iterations is reached. The code also generates a plot of the best and average fitness values for each generation. The user can select a dataset and input parameters such as population size, gene number, and mutation probability.

### 🙌 All together
📽️ Refer this video for watching whole simulation on
<a href="https://youtu.be/xVEqZ3lY2Mk" target="_blank">YouTube.</a>

### Collaboration
Collaborated with [Alperen Ölçer](https://github.com/Alperenlcr)