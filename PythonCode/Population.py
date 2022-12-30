import numpy as np
from Agent import Agent
import random
from AgentUtilities import *
import os
from PathManipulator import PathManipulator
from TestUtilFunctions import has_duplicates


class Population:
    size: int
    age: int
    cities: np.array
    pathManipulator : PathManipulator
    agents: list[Agent]
    amount_of_permutations: int
    probability_of_permutation: float
    k: int
    bestAgent : Agent
    # Initiate Population with size, graph and cities
    def __init__(self, population_size, graph, cities):
        self.size = population_size
        self.age = 0
        self.cities = cities
        self.pathManipulator = PathManipulator(graph)
        self.agents = [Agent(0, self.pathManipulator.shortest_n(self.pathManipulator.original[0].size//5)) for _ in range(population_size)]
        #self.agents = [Agent(0,np.array(random.sample(cities,len(cities)))) for _ in range(population_size)]
        for agent in self.agents:
            self.pathManipulator.remove_infinite_path_closest_city(agent.cities)
            self.pathManipulator.hard_remove_infs(agent.cities)

        # TODO: Fix this.
        self.amount_of_permutations = len(cities)//10
        self.probability_of_permutation = 0.3
        self.k = 5#25
        self.bestAgent = None

    # # Select top x agents with the highest fitness
    # def select(self):
    #     self.agents.sort(key=lambda agent: evaluateAgent(agent, self.graph), reverse=True)
    #     self.agents = self.agents[self.size // 3:]  # take top x agents (eg: population size/3)

    # Mutate each agent
    def mutate(self):
        for agent in self.agents:
            if os.environ["MUTATE"] == "RANDOMSWAP":
                agent.mutate(self.amount_of_permutations, self.probability_of_permutation)
            elif os.environ["MUTATE"] == "LOCALSEARCH":
                agent.mutateLocalSearch(self.pathManipulator.copy,5)
            elif os.environ["MUTATE"] == "3-OPT":
                agent.three_opt(self.pathManipulator.copy)
            elif os.environ["MUTATE"] == "RANDOMSWAPELIMINFS":
                agent.mutateEliminateInfs(self.amount_of_permutations,self.probability_of_permutation,self.pathManipulator)
                agent.cities = self.pathManipulator.hard_remove_infs(agent.cities)
                agent.hasBeenModified = True
            elif os.environ["MUTATE"] == "RANDOMOPTIMIZED":
                agent.mutate(self.amount_of_permutations,self.probability_of_permutation)
                agent.cities = self.pathManipulator.two_opt_alt(agent.cities,
                                                            agent.evaluateAgent(self.pathManipulator.copy))

            else:
                raise ValueError("No Mutation Algorithm selected")

    # Crossover agents selected by top k tournament selection
    def crossover(self):
        children = []
        if os.environ["CROSSOVER"] == "DEFAULT":
            for i in range(self.size//2):
                parent1 = self.kTournamentSelection()
                parent2 = self.kTournamentSelection()
                child = crossoverAgents(parent1, parent2, self.age)

                children.append(child)
        elif os.environ["CROSSOVER"] == "ORDERED":
            for i in range(self.size//2):
                parent1 = self.kTournamentSelection()
                parent2 = self.kTournamentSelection()
                child1,child2 = ordered_crossover(parent1.cities, parent2.cities, self.age)

                self.agents.append(child1)
                self.agents.append(child2)
        elif os.environ["CROSSOVER"] == "CYCLE2":
            for i in range(self.size//4):
                parent1 = self.kTournamentSelection()
                parent2 = self.kTournamentSelection()
                child1,child2 = CX2_crossover(parent1.cities, parent2.cities, self.age)

                children.append(child1)
                children.append(child2)
        elif os.environ["CROSSOVER"] == "OPTEXCLUSIVITY":
            selected_agents = []
            for _ in range(self.size//2):
                parent1 = self.kTournamentSelection()
                while parent1 in selected_agents:
                    parent1 = self.kTournamentSelection()
                selected_agents.append(parent1)
                parent1.cities = self.pathManipulator.two_opt_alt(parent1.cities,parent1.evaluateAgent(self.pathManipulator.copy))
                for _ in range(self.size//10):
                    parent2 = self.kTournamentSelection()
                    child1,child2 =ordered_crossover(parent1.cities, parent2.cities, self.age)
                    self.agents.append(child1)
                    self.agents.append(child2)
        self.agents = self.agents + children





    def eliminate(self):
        while len(self.agents) > self.size:
            self.agents.remove(self.kTournamentElimination())

    def kTournamentElimination(self):
        agent_sample = random.sample(self.agents, self.k)
        fitness_scores = [agent.evaluateAgent(self.pathManipulator.copy) for agent in agent_sample]
        if (os.getenv("ELIMINATION") == "DEFAULT"):
            return self.agents[self.agents.index(agent_sample[fitness_scores.index(max(fitness_scores))])]
        elif os.getenv("ELIMINATION") == "FITNESSSHARING":
            sharing_fitness_score = self.pathManipulator.fitness_sharing(np.array(fitness_scores),2)
            idx1 = np.where(sharing_fitness_score==max(sharing_fitness_score))
            sampledAgent = agent_sample[idx1[0][0]]
            return self.agents[self.agents.index(sampledAgent)]

    def kTournamentSelection(self):
        agent_sample = random.sample(self.agents, self.k)
        fitness_scores = [agent.evaluateAgent(self.pathManipulator.copy) for agent in agent_sample]
        if os.getenv("SCORE") == "DEFAULT":
            return self.agents[self.agents.index(agent_sample[fitness_scores.index(min(fitness_scores))])]
        elif os.getenv("SCORE") == "FITNESSSHARING":
            sharing_fitness_score = self.pathManipulator.fitness_sharing(np.array(fitness_scores),2)
            idx1 = np.where(sharing_fitness_score==min(sharing_fitness_score))
            sampledAgent = agent_sample[idx1[0][0]]
            return self.agents[self.agents.index(sampledAgent)]


    def getObjectiveValues(self):
        fitness_scores = [agent.evaluateAgent(self.pathManipulator.original) for agent in self.agents]
        min_score = min(fitness_scores)
        avg_score = sum(fitness_scores) / len(fitness_scores)
        bestSolution = self.agents[fitness_scores.index(min(fitness_scores))]
        if not bestSolution.isBestAgent and self.bestAgent is None:
            bestSolution.isBestAgent = True
        elif not bestSolution.isBestAgent and self.bestAgent.score > bestSolution.score:
            self.bestAgent.isBestAgent = False
            self.bestAgent = bestSolution
            bestSolution.isBestAgent = True
        elif not bestSolution.isBestAgent and self.bestAgent.score < bestSolution.score:
            min_score = self.bestAgent.score
            bestSolution = self.bestAgent

        return min_score, avg_score, bestSolution.cities

    def resetAgentScores(self):
        for agent in self.agents:
            agent.hasBeenModified = True


    def applyTwoOpt(self):
        for agent in self.agents:
            #agent.cities = self.pathManipulator.two_opt(agent.cities,agent.evaluateAgent(self.pathManipulator.copy))
            agent.cities = self.pathManipulator.two_opt_alt_alt(agent.cities,agent.evaluateAgent(self.pathManipulator.copy))

    def applyTwoOptRandomized(self):
        for agent in self.agents:
            agent.cities = self.pathManipulator.two_opt_ranomized(agent.cities,agent.evaluateAgent(self.pathManipulator.copy))

    def applyTwoOptRandomSubset(self):
        arr = []
        for _ in range(self.size//10):
            rand = random.randrange(0,self.size)
            while rand in arr:
                rand = random.randrange(0,self.size)
            arr.append(arr)
            currAgent = self.agents[rand]
            currAgent.cities = self.pathManipulator.two_opt_alt(currAgent.cities,currAgent.evaluateAgent(self.pathManipulator.copy))
