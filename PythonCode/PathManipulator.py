import random

import numpy as np


class PathManipulator:

    original : np.array
    copy : np.array

    def __init__(self,distanceMatrix):
        self.original = distanceMatrix
        self.copy = np.copy(distanceMatrix)

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
                for j in range(1, cities.size - 1):
                    if j != i and j != i + 1 and self.copy[cities[i]][cities[j]] < min_distance \
                            and self.copy[cities[j - 1]][cities[i + 1]] != np.inf \
                            and self.copy[cities[i + 1]][cities[j + 1]] != np.inf:
                        min_distance = self.copy[cities[i]][cities[j]]
                        min_idx = j

                if min_idx != -1:
                    cities[i+1],cities[min_idx] = cities[min_idx],cities[i+1]
                else:
                    raise ValueError("No suitable element found to swap with")

        return cities

    def introduce_infinities(self,amountOfInfinities):
        self.copy = np.copy(self.original)

        for _ in range(amountOfInfinities):
            citya = random.randrange(0,self.original[0].size)
            cityb = random.randrange(0,self.original[0].size)
            while cityb == citya and self.original[citya][cityb] != np.inf:
                cityb = random.randrange(0, self.original[0].size)

            self.copy[citya][cityb] = np.inf