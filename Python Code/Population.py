import numpy as np
from Agent import Agent
import random
from AgentUtilities import *


class Population:
    size: int
    age: int
    cities: list[int]
    graph: np.array
    agents: list[Agent]
    amount_of_permutations: int
    probability_of_permutation: float

    # Initiate Population with size, graph and cities
    def __init__(self, population_size, graph, cities):
        self.size = population_size
        self.age = 0
        self.cities = cities
        self.graph = graph
        self.agents = [Agent(0, random.sample(cities, len(cities))) for i in range(population_size)]
        self.amount_of_permutations = len(cities) // 2
        self.probability_of_permutation = 0.5

    # Select top x agents with the highest fitness
    def select(self):
        self.agents.sort(key=lambda agent: evaluateAgent(agent, self.graph), reverse=True)
        self.agents = self.agents[self.size // 3:]  # take top x agents (eg: population size/3)

    # Mutate each agent
    def mutate(self):
        for agent in self.agents:
            agent.mutate(self.amount_of_permutations, self.probability_of_permutation)

    # Crossover agents selected by top k tournament selection
    def crossover(self):
        self.topKTournamentSelection()
        crossoverAgents( None, None, self.age)

    def topKTournamentSelection(self):
        pass
