import random

import numpy


class Agent:
    cities : list[int]
    age : int


    # Initiate Agent with age and list of cities in order of visitation.
    #
    def __init__(self, age, cities):
        self.cities = cities
        self.age = age

    # Function that'll change the internal cities array depending on 2 parameters
    # 1. Amount of permutations: Represents the amount of random switches within the cities array
    # 2. Probability of Permutations: if random no. between [0;1] is smaller than prob. then switch will occur.
    def mutate(self, amount_of_permutations, probability_of_permutation):
        for i in range(amount_of_permutations):
            if random.random() < probability_of_permutation:
                randa = random.randrange(len(self.cities))
                randb = random.randrange(len(self.cities))
                self.cities[randa], self.cities[randb] = self.cities[randb], self.cities[randa]


