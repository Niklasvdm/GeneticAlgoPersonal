from Agent import Agent
import numpy as np
import random
import math


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