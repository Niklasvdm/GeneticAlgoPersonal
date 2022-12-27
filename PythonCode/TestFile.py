import PathManipulator
from Population import Population
import numpy as np
from TestUtilFunctions import has_duplicates,fitness_sharing
from AgentUtilities import mutateLocalSearch

file = open("/home/Universiteit/GeneticAlgorithms/tours/tour50.csv")
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()
cities = list(range(distanceMatrix[0].size))
population = Population(400, distanceMatrix, cities)

def verifyInf():
    for agent in population.agents:
        #assert(not population.pathManipulator.has_infinite_path(agent.cities))
        if (population.pathManipulator.has_infinite_path(agent.cities)):
            print(agent.cities)
            rand : [] = []
            for i in range(len(agent.cities)-1):
                rand.append(population.pathManipulator.copy[agent.cities[i]][agent.cities[i+1]])
            print(rand)



arr = np.array([i for i in range(50)])
mutateLocalSearch(arr,population.pathManipulator.copy,5,0)
print(arr)


for _ in range(50):
    print(population.pathManipulator.shortest_n(5))
    continue
    print("HELLO")
    population.crossover()
    population.mutate()
    population.eliminate()

    for agent in population.agents:
        if has_duplicates(agent.cities):
            print("FUCK YOU!!")
            print(agent.cities)
