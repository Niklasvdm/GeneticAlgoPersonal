from Agent import Agent
import numpy as np
import random
import math
from TestUtilFunctions import has_duplicates


# evaluateAgent will return the length of the cycle the agent has.
# length(pi) = distance(cities[n-1],cities[0]) + sum_(i=0)^(i=n-1) [ distance(cities[i],cities[i+1] ]
def evaluateAgent(agent, cityGraph):
    i = 0
    pathLength = 0
    cities: np.array = agent.cities
    while i != len(cities) - 1:
        city_0 = cities[i]
        city_1 = cities[i + 1]

        if not math.isinf(cityGraph[city_0][city_1]):
            pathLength += cityGraph[city_0][city_1]
        else:
            pathLength *= 2
        i += 1
    pathLength += cityGraph[i][0]

    return pathLength


# crossoverAgents will combine two agents into a new randomly created agent as a combination of both.
# Parameters: agent_1 and agent_2, the two parents of the newly created agent
#           | Age: the current cycle of the generational Algorithm, will be passed down onto the newly created agent.
# Returns: The newly created agent.
#
def crossoverAgents(agent_1, agent_2, age):
    # new_cities = agent_1.cities
    #
    # diff_idx = [idx for idx, element in enumerate(agent_1.cities) if agent_2.cities[idx] != agent_1.cities[idx]]
    # if len(diff_idx) >= 2:
    #     for i in range(100):
    #         random_idx_pair = random.sample(diff_idx, 2)
    #         random_idx_pair[0], random_idx_pair[1] = random_idx_pair[1], random_idx_pair[0]
    # return Agent(age, new_cities)

    new_cities = agent_1.cities
    p2_cities = agent_2.cities
    subset_start = random.randrange(0, len(new_cities)-1)
    subset_length = random.randrange(1, len(new_cities)-subset_start)

    subset = new_cities[subset_start:subset_start+subset_length]
    new_positions = random.sample(range(0, len(new_cities)), subset_length)
    new_positions.sort()

    subset_index = 0

    new_cities = [-1] * len(p2_cities)

    for i in range(len(new_cities)):
        if i in new_positions:
            new_cities[i] = subset[subset_index]
            subset_index+=1
        else:
            for city in agent_2.cities:
                if city not in subset and city not in new_cities:
                    new_cities[i] = city
                    break

    return Agent(age, new_cities)



def ordered_crossover(cities1: np.array, cities2: np.array,age : int) -> tuple[Agent, Agent]:
        # Choose a random start and end index for the window
        start = np.random.randint(cities1.size)
        end = np.random.randint(start, cities1.size)

        # Extract the window from the first parent
        window = cities1[start:end + 1]

        # Initialize the offspring as copies of the second parent
        offspring1 = cities2.copy()
        offspring2 = cities2.copy()

        # Insert the window into the offspring
        offspring1[start:end + 1] = window
        offspring2[start:end + 1] = window

        # Fill in the remaining elements from the second parent
        i = 0
        for city in cities2:
            if city not in window:
                if i == start:
                    i = end + 1
                offspring1[i] = city
                i += 1

        i = 0
        for city in cities1:
            if city not in window:
                if i == start:
                    i = end + 1
                offspring2[i] = city
                i += 1

        # Verify that the offspring do not contain any duplicate elements
        #if has_duplicates(offspring1) or has_duplicates(offspring2):
        #    raise ValueError("Offspring contain duplicate elements")

        return Agent(age,offspring1), Agent(age,offspring2)


def CX2_crossover(cities1: np.array, cities2: np.array,age : int) -> tuple[Agent, Agent]:
    # Initialize the offspring as copies of the parents
    offspring1 = cities1.copy()
    offspring2 = cities2.copy()

    # Initialize the visited arrays
    visited1 = np.zeros_like(cities1, dtype=bool)
    visited2 = np.zeros_like(cities2, dtype=bool)

    # Iterate over the elements of the parents
    for i in range(cities1.size):
        # Skip visited elements
        if visited1[i] or visited2[i]:
            continue

        # Initialize the cycles
        cycle1 = [i]
        cycle2 = [i]

        # Follow the cycles until they come back to the starting element
        j = i
        while not visited1[j]:
            visited1[j] = True
            j = np.where(cities1 == cities2[j])[0][0]
            cycle1.append(j)

        j = i
        while not visited2[j]:
            visited2[j] = True
            j = np.where(cities2 == cities1[j])[0][0]
            cycle2.append(j)

        # Swap the cycles between the offspring
        offspring1[cycle1] = cities2[cycle2]
        offspring2[cycle2] = cities1[cycle1]

    # # Verify that the offspring do not contain any duplicate elements
    # if has_duplicates(offspring1) or has_duplicates(offspring2):
    #     raise ValueError("Offspring contain duplicate elements")

    return Agent(age,offspring1), Agent(age,offspring2)


def mutateLocalSearch(cities : np.array, distanceMatrix : np.array, depth : int , age: int):
    city = cities[0]

    for i in range(depth):
        closestcity = cities[i+1]
        currentDistance = distanceMatrix[city][cities[1]]
        for j in range(len(distanceMatrix[city])):
            if j == i:
                continue

            if distanceMatrix[city][j] < currentDistance:
                currentDistance = distanceMatrix[city][j]
                closestcity = j

        idxOfClosestCity = np.where(cities == j)
        cities[i+1],cities[idxOfClosestCity[0]] = cities[idxOfClosestCity[0]],cities[i+1]



