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
        i += 1
    pathLength += cityGraph[i][0]

    return pathLength


# crossoverAgents will combine two agents into a new randomly created agent as a combination of both.
# Parameters: agent_1 and agent_2, the two parents of the newly created agent
#           | Age: the current cycle of the generational Algorithm, will be passed down onto the newly created agent.
# Returns: The newly created agent.
#
def crossoverAgents(agent_1, agent_2, age):
    new_cities = agent_1.cities

    diff_idx = [idx for idx, element in enumerate(agent_1.cities) if agent_2.cities[idx] != agent_1.cities[idx]]
    if len(diff_idx) >= 2:
        for i in range(100):
            random_idx_pair = random.sample(diff_idx, 2)
            random_idx_pair[0], random_idx_pair[1] = random_idx_pair[1], random_idx_pair[0]
    return Agent(age, new_cities)


# array1 = np.array([i for i in range(10)])
# array2 = array1.copy()
# random.shuffle(array2)
# randomAgent1 = Agent(0, array1)
# randomAgent2 = Agent(0, array2)
# randomAgent3 = crossoverAgents(randomAgent1, randomAgent2, 1)
# print("First Agent array", randomAgent1.cities)
# print("Second Agent array", randomAgent2.cities)
# print("Third Agent array", randomAgent3.cities)
