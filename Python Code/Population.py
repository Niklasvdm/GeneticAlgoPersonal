import numpy
from Agent import Agent
import random


class Population:
    size: int
    agents: list[int]

    # Initiate Agent with age and list of cities in order of visitation.
    #
    def __init__(self, population_size, graph, cities):
        self.agents = [Agent(0, random.shuffle(cities)) for i in range(population_size)]
        self.size = population_size
        self.cities = cities


    def mutatePopulation(self):
        for agent in self.agents:
            agent.mutate()

    def crossoverPopulation(self):
        pass

    def eliminatePopulation(self):
        pass
