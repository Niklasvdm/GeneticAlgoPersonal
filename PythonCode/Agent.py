import random

import numpy as np
import PathManipulator
from math import isinf


class Agent:
    cities: list[int]
    age: int
    score : int
    hasBeenModified : bool

    # Initiate Agent with age and list of cities in order of visitation.
    #
    def __init__(self, age, cities):
        self.cities = cities
        self.age = age
        self.score = 0
        self.hasBeenModified = True

    # Function that'll change the internal cities array depending on 2 parameters
    # 1. Amount of permutations: Represents the amount of random switches within the cities array
    # 2. Probability of Permutations: if random no. between [0;1] is smaller than prob. then switch will occur.
    def mutate(self, amount_of_permutations, probability_of_permutation):
        for i in range(amount_of_permutations):
            randomInt = random.random()
            #print("The random int generated is: ", randomInt , " And the prob. of permutation is: " , probability_of_permutation)
            if randomInt < probability_of_permutation:
                randa = random.randrange(len(self.cities))
                randb = random.randrange(len(self.cities))
                self.cities[randa], self.cities[randb] = self.cities[randb], self.cities[randa]
        self.hasBeenModified = True


    def mutateLocalSearch(self, distanceMatrix : np.array, depth : int):
        city = self.cities[0]

        for i in range(depth):
            closestcity = self.cities[i+1]
            currentDistance = distanceMatrix[city][closestcity]
            for j in range(len(distanceMatrix[city])):
                if j == i:
                    continue

                if distanceMatrix[city][j] < currentDistance:
                    currentDistance = distanceMatrix[city][j]
                    closestcity = j

            idxOfClosestCity = np.where(self.cities == closestcity)
            self.cities[i+1],self.cities[idxOfClosestCity[0]] = self.cities[idxOfClosestCity[0]],self.cities[i+1]



    #def three_opt(self, distanceMatrix: np.array):
        # Set the improvement threshold
    #    improvement_threshold = 1e-6

        # Iterate until no further improvement is possible
    #    while True:
    #        # Initialize the best solution and distance as the current solution and distance
    #        best_solution = self.cities
    #        best_distance = evaluateAgent(self, distanceMatrix)

            # # Iterate over the possible changes
            # for i in range(self.cities.size):
            #     for j in range(i + 1, self.cities.size):
            #         for k in range(j + 1, self.cities.size):
            #             # Add the three cities to the tour
            #             candidate_solution = np.insert(self.cities, i + 1, self.cities[j:k + 1])
            #             candidate_distance = evaluateAgent(self, distanceMatrix)
            #             if candidate_distance < best_distance:
            #                 best_solution = candidate_solution
            #                 best_distance = candidate_distance
            #
            #             # Delete the three cities from the tour
            #             candidate_solution = np.delete(self.cities, np.s_[i + 1:k + 1])
                    #     candidate_distance = evaluateAgent(self, distanceMatrix)
                    #     # Reverse the three cities in the tour
                    #     candidate_solution = np.concatenate([self.cities[:i + 1], self.cities[j:k + 1][::-1], self.cities[k + 1:]])
                    #     candidate_distance = evaluateAgent(self, distanceMatrix)
                    #     if candidate_distance < best_distance:
                    #         best_solution = candidate_solution
                    #         best_distance = candidate_distance
                    #
                    #     # Check if the best solution represents an improvement over the current solution
                    # improvement = evaluateAgent(self, distanceMatrix) - best_distance
                    # if improvement < improvement_threshold:
                    #     break

                    # Update the current solution
                    # self.cities = best_solution

    def evaluateAgent(self,distanceMatrix : np.array) -> int:
        if not self.hasBeenModified:
            return self.score
        else:
            i = 0
            pathLength = 0
            cities: np.array = self.cities
            while i != len(cities) - 1:
                city_0 = cities[i]
                city_1 = cities[i + 1]

                if not isinf(distanceMatrix[city_0][city_1]):
                    pathLength += distanceMatrix[city_0][city_1]
                else:
                    pathLength *= 2
                i += 1
            pathLength += distanceMatrix[i][0]

            self.score = pathLength
            self.hasBeenModified = False
            return pathLength


    def mutateEliminateInfs(self, amount_of_permutations, probability_of_permutation,pathManipulator : PathManipulator):
        for i in range(amount_of_permutations):
            randomInt = random.random()
            #print("The random int generated is: ", randomInt , " And the prob. of permutation is: " , probability_of_permutation)
            if randomInt < probability_of_permutation:
                randa = random.randrange(len(self.cities))
                randb = random.randrange(len(self.cities))
                self.cities[randa], self.cities[randb] = self.cities[randb], self.cities[randa]

        self.cities = pathManipulator.remove_infinite_path_closest_city(self.cities)
        self.hasBeenModified = True