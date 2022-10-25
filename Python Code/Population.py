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
    k: int

    # Initiate Population with size, graph and cities
    def __init__(self, population_size, graph, cities):
        self.size = population_size
        self.age = 0
        self.cities = cities
        self.graph = graph
        self.agents = [Agent(0, random.sample(cities, len(cities))) for i in range(population_size)]
        self.amount_of_permutations = len(cities) // 2
        self.probability_of_permutation = 0.8
        self.k = 5

    # # Select top x agents with the highest fitness
    # def select(self):
    #     self.agents.sort(key=lambda agent: evaluateAgent(agent, self.graph), reverse=True)
    #     self.agents = self.agents[self.size // 3:]  # take top x agents (eg: population size/3)

    # Mutate each agent
    def mutate(self):
        for agent in self.agents:
            agent.mutate(self.amount_of_permutations, self.probability_of_permutation)

    # Crossover agents selected by top k tournament selection
    def crossover(self):
        for i in range(self.size // 2):
            parent1 = self.kTournamentSelection()
            parent2 = self.kTournamentSelection()
            child = crossoverAgents(parent1, parent2, self.age)

            self.agents.append(child)

    def eliminate(self):
        while len(self.agents) > self.size:
            self.agents.remove(self.kTournamentElimination())

    def kTournamentElimination(self):
        agent_sample = random.sample(self.agents, self.k)
        fitness_scores = [evaluateAgent(agent, self.graph) for agent in agent_sample]
        return self.agents[fitness_scores.index(max(fitness_scores))]

    def kTournamentSelection(self):
        agent_sample = random.sample(self.agents, self.k)
        fitness_scores = [evaluateAgent(agent, self.graph) for agent in agent_sample]
        return self.agents[fitness_scores.index(min(fitness_scores))]

    def getObjectiveValues(self):
        fitness_scores = [evaluateAgent(agent, self.graph) for agent in self.agents]
        min_score = min(fitness_scores)
        avg_score = sum(fitness_scores) / len(fitness_scores)
        bestSolution = self.agents[fitness_scores.index(min(fitness_scores))].cities

        return min_score, avg_score, bestSolution
