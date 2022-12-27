import random

import numpy
import numpy as np


class PathManipulator:

    original : np.array
    copy : np.array
    closestCity : np.array

    def __init__(self,distanceMatrix):
        self.original = distanceMatrix
        self.copy = np.copy(distanceMatrix)
        self.closestCity = np.full(self.original[0].size , -1)

    def eval_agent(self,cities : np.array) -> int:
        i = 0
        pathLength = 0
        while i != len(cities) - 1:
            city_0 = cities[i]
            city_1 = cities[i + 1]

            if not self.original[city_0][city_1] == np.inf:
                pathLength += self.original[city_0][city_1]
            else:
                pathLength *= 2
            i += 1
        pathLength += self.original[i][0]

        return pathLength

    def has_infinite_path(self,cities: np.array) -> bool:
        # Check if the array has at least two elements
        if cities.size < 2:
            return False

        # Check if any two consecutive elements in the array have an infinite distance between them
        for i in range(cities.size - 1):
            if self.copy[cities[i]][cities[i + 1]] == np.inf:
                return True

        # If none of the above conditions are met, return False
        return False

    def remove_infinite_path(self,cities: np.array) -> np.array:
        # Check if the array has at least two elements
        if len(cities) < 2:
            return cities

        # Check if any two consecutive elements in the array have an infinite distance between them
        for i in range(len(cities) - 1):
            if self.copy[cities[i]][cities[i + 1]] == np.inf:
                # Find another element to swap with
                for j in range(1,cities.size - 1):
                    if j != i and j != i + 1 and self.copy[cities[i]][cities[j]] != np.inf \
                            and self.copy[cities[j-1]][cities[i+1]] != np.inf and self.copy[cities[i+1]][cities[j+1]] != np.inf:
                        cities[i + 1],cities[j] = cities[j],cities[i+1]
                        break
                else:
                    raise ValueError("No suitable element found to swap with")

        return cities

    def remove_infinite_path_closest_city(self,cities:np.array) -> np.array:
        # Check if the array has at least two elements
        if len(cities) < 2:
            return cities

        # Check if any two consecutive elements in the array have an infinite distance between them
        for i in range(len(cities) - 1):
            if self.copy[cities[i]][cities[i + 1]] == np.inf:
                # Find another element to swap with
                min_distance = np.inf
                min_idx = -1

                if self.closestCity[cities[i]] == -1:
                    for j in range(1, cities.size - 1):
                        if j != i and j != i + 1 and self.copy[cities[i]][cities[j]] < min_distance \
                                and self.copy[cities[j - 1]][cities[i + 1]] != np.inf \
                                and self.copy[cities[i + 1]][cities[j + 1]] != np.inf:
                            min_distance = self.copy[cities[i]][cities[j]]
                            min_idx = j
                else:

                    idxOfClosestCity = int(np.where(self.closestCity[cities[i]] == cities)[0][0])
                    if  self.closestCity[cities[i]] != 1 and self.copy[cities[idxOfClosestCity - 1]][cities[i + 1]] != np.inf \
                    and self.copy[cities[i + 1]][cities[(idxOfClosestCity + 1)%cities.size]] != np.inf:
                        min_idx = idxOfClosestCity



                if min_idx != -1:
                    cities[i+1],cities[min_idx] = cities[min_idx],cities[i+1]
                # else:
                #     raise ValueError("No suitable element found to swap with")

        return cities

    def introduce_infinities(self,amountOfInfinities):
        self.copy = np.copy(self.original)

        for _ in range(amountOfInfinities):
            citya = random.randrange(0,self.original[0].size)
            cityb = random.randrange(0,self.original[0].size)
            while cityb == citya and self.original[citya][cityb] != np.inf:
                cityb = random.randrange(0, self.original[0].size)

            self.copy[citya][cityb] = np.inf

    def fitness_sharing(self,fitness: np.array, sigma: float) -> np.array:
        # Calculate the Jaccard distance matrix
        distance_matrix = np.zeros((fitness.size, fitness.size))
        for i in range(fitness.size):
            for j in range(i + 1, fitness.size):
                intersection = np.intersect1d(fitness[i], fitness[j]).size
                union = np.union1d(fitness[i], fitness[j]).size
                distance_matrix[i, j] = distance_matrix[j, i] = 1 - intersection / union

        # Calculate the shared fitness values
        shared_fitness = np.zeros(fitness.size)
        for i in range(fitness.size):
            sum_share = 0
            for j in range(fitness.size):
                if i != j:
                    sum_share += np.exp(-distance_matrix[i, j] ** 2 / (2 * sigma ** 2))
            shared_fitness[i] = fitness[i] / sum_share

        return shared_fitness


    def shortest_n(self,depth : int) -> np.array:
        cities = np.full(depth+1,-1)
        cities[0] = random.randrange(0,self.original[0].size)
        unusedIndexes = [i for i in range(self.original[0].size)]
        for i in range(depth):
            cityIdx = cities[i]
            unusedIndexes.remove(cityIdx)

            if self.closestCity[cityIdx] != -1:
                if not np.any(cities==self.closestCity[cityIdx]):
                    cities[i+1] = self.closestCity[cityIdx]
                    continue

            closest_city = -1
            closest_distance = np.inf
            for j in range(self.original[i].size):
                if (cityIdx != j and self.original[cities[i]][j] < closest_distance and not np.any(cities == j)):
                    closest_city = j
                    closest_distance = self.original[cityIdx][j]
            cities[i+1] = closest_city
            self.closestCity[cityIdx] = closest_city
        unusedIndexes.remove(cities[depth])
        unusedIndexes = np.array(unusedIndexes)
        np.random.shuffle(unusedIndexes)

        return numpy.concatenate((cities , unusedIndexes),axis=0)

    # Code inspired by https://github.com/pdrm83/py2opt/blob/master/py2opt/solver.py
    def two_opt(self, cities : np.array,heuristic : int, improvement_threshold=0.01) -> np.array:
        copy = np.copy(cities)
        best_distance = heuristic

        #self.best_route = self.initial_route
        #self.best_distance = self.calculate_path_dist(self.distance_matrix, self.best_route)
        improvement_factor = 1

        while improvement_factor > improvement_threshold:
            previous_best = best_distance
            for swap_first in range(1, cities.size - 2):
                for swap_last in range(swap_first + 1, cities.size - 1):
                    before_start = copy[swap_first - 1]
                    start = copy[swap_first]
                    end = copy[swap_last]
                    after_end = copy[swap_last + 1]
                    before = self.original[before_start][start] + self.original[end][after_end]
                    after = self.original[before_start][end] + self.original[start][after_end]
                    if after < before:
                        new_route = self.swap(copy, swap_first, swap_last)
                        new_distance = self.eval_agent(new_route)

                        copy = new_route
                        best_distance = new_distance

            improvement_factor = 1 - best_distance / previous_best
        return copy


    @staticmethod
    def swap(path, swap_first, swap_last):
        path_updated = np.concatenate((path[0:swap_first],
                                       path[swap_last:-len(path) + swap_first - 1:-1],
                                       path[swap_last + 1:len(path)]))
        return path_updated