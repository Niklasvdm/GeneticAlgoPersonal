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
#print(arr)


for agent in population.agents:
    print(agent.cities)
    cities1 = population.pathManipulator.two_opt(agent.cities,agent.evaluateAgent(population.pathManipulator.copy))
    print(cities1 , population.pathManipulator.eval_agent(cities1)," contains duplicats: ", has_duplicates(cities1), " has infs: ", population.pathManipulator.has_infinite_path(cities1))
    cities2 = population.pathManipulator.two_opt_alt(agent.cities, agent.evaluateAgent(population.pathManipulator.copy))
    print(cities2, population.pathManipulator.eval_agent(cities2),"Contains Duplicates: ", has_duplicates(cities2), " has infs: ", population.pathManipulator.has_infinite_path(cities2))
    print (list([elem1 == elem2 for (elem1,elem2) in zip(agent.cities,cities2)]))
