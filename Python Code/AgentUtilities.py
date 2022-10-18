import Agent
import numpy


# evaluateAgent will return the length of the cycle the agent has.
# length(pi) = distance(cities[n-1],cities[0]) + sum_(i=0)^(i=n-1) [ distance(cities[i],cities[i+1] ]
def evaluateAgent(agent, cityGraph):
    i = 0
    pathLength = 0
    cities : numpy.array = agent.cities
    while i != cities.size - 1:
        city_0 = cities[i]
        city_1 = cities[i+1]

        pathLength += cityGraph[city_0][city_1]
        i += 1
    pathLength += cityGraph[i][0]

    return pathLength


# crossoverAgents will combine two agents into a new randomly created agent as a combination of both.
# Parameters: agent_1 and agent_2, the two parents of the newly created agent
#           | Age: the current cycle of the generational Algorithm, will be passed down onto the newly created agent.
# Returns: The newly created agent.
#
def crossoverAgents(agent_1,agent_2,age): #Returns new Agent.

