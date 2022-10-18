import numpy


class Agent:
    cities : numpy.array
    age : int


    # Initiate Agent with age and list of cities in order of visitation.
    #
    def __init__(self, age, cities):
        self.cities = cities
        self.age = age

    # Function that'll change the internal cities array depending on 2 parameters
    # 1. Amount of permutations: Represents the amount of random switches within the cities array
    # 2. Probability of Permutations: if random no. between [0;1] is bigger than prob. then switch will occur.
    def mutate(self, amount_of_permutations, probability_of_permutation):
        permutated_cities = self.cities


