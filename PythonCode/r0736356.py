import Reporter
import numpy as np
import random
from math import isinf





class PathManipulator:
    original : np.array
    copy : np.array
    closestCity : np.array

    def __init__(self,distanceMatrix):
        self.original = distanceMatrix
        self.copy = np.copy(distanceMatrix)
        self.closestCity = np.full(self.original[0].size , -1)

    def eval_agent(self,cities : np.array) -> int:
        i = 0
        pathLength = 0
        while i != len(cities) - 1:
            city_0 = cities[i]
            city_1 = cities[i + 1]

            if not self.original[city_0][city_1] == np.inf:
                pathLength += self.original[city_0][city_1]
            else:
                pathLength += 100000
                #print("Shpuldn't happen, Eval_Agent")
            i += 1
        pathLength += self.original[cities[i]][cities[0]]

        return pathLength


    def eval_subpath(self,cities : np.array) -> int:
        i = 0
        pathLength = 0
        while i != len(cities) - 1:
            city_0 = cities[i]
            city_1 = cities[i + 1]

            if not self.original[city_0][city_1] == np.inf:
                pathLength += self.original[city_0][city_1]
            else:
                pathLength += 100000
                #print("Shouldn't happen, eval subpath")
            i += 1

        return pathLength

    def has_infinite_path(self,cities: np.array) -> bool:
        # Check if the array has at least two elements
        if cities.size < 2:
            return False

        # Check if any two consecutive elements in the array have an infinite distance between them
        for i in range(cities.size -1):
            if self.copy[cities[i]][cities[i + 1]] == np.inf:
                return True
        if(self.copy[cities[-1]][cities[0]]) == np.inf:
            return True

        # If none of the above conditions are met, return False
        return False


    def remove_infinite_path_closest_city(self,cities:np.array) -> np.array:
        # Check if the array has at least two elements
        if len(cities) < 2:
            return cities

        # Check if any two consecutive elements in the array have an infinite distance between them
        for i in range(len(cities) - 1):
            if self.copy[cities[i]][cities[i + 1]] == np.inf:
                # Find another element to swap with
                min_distance = np.inf
                min_idx = -1

                if self.closestCity[cities[i]] == -1:
                    for j in range(1, cities.size - 1):
                        if j != i and j != i + 1 and self.copy[cities[i]][cities[j]] < min_distance \
                                and self.copy[cities[j - 1]][cities[i + 1]] != np.inf \
                                and self.copy[cities[i + 1]][cities[j + 1]] != np.inf:
                            min_distance = self.copy[cities[i]][cities[j]]
                            min_idx = j
                else:

                    idxOfClosestCity = int(np.where(self.closestCity[cities[i]] == cities)[0][0])
                    if  self.closestCity[cities[i]] != 1 and self.copy[cities[idxOfClosestCity - 1]][cities[i + 1]] != np.inf \
                    and self.copy[cities[i + 1]][cities[(idxOfClosestCity + 1)%cities.size]] != np.inf:
                        min_idx = idxOfClosestCity



                if min_idx != -1:
                    cities[i+1],cities[min_idx] = cities[min_idx],cities[i+1]
                # else:
                #     raise ValueError("No suitable element found to swap with")

        return cities


    def hard_remove_infs(self,cities : np.array) -> np.array:
        while self.eval_agent(cities) == np.inf:
            for i in range(len(cities) - 1):
                if self.copy[cities[i]][cities[i + 1]] == np.inf:
                    # Find another element to swap with
                    for j in range(1, cities.size - 1):
                        if j != i and j != i + 1 and self.copy[cities[i]][cities[j]] != np.inf \
                                and self.copy[cities[j - 1]][cities[i + 1]] != np.inf \
                                and self.copy[cities[i + 1]][cities[j + 1]] != np.inf:
                            cities[i + 1], cities[j] = cities[j], cities[i + 1]
                            break
                    else:
                        raise ValueError("No suitable element found to swap with")
            if (self.copy[cities[-1]][cities[0]]) == np.inf:
                for i in range(1, cities.size - 2):
                    if self.copy[cities[i]][cities[0]] != np.inf \
                            and self.copy[cities[-2]][cities[i]] != np.inf \
                            and self.copy[cities[i - 1]][cities[-1]] != np.inf \
                            and self.copy[cities[i + 1]][cities[-1]] != np.inf:
                        cities[i], cities[-1] = cities[-1], cities[i]
                        break

        return cities


    def introduce_infinities(self,amountOfInfinities):
        self.copy = np.copy(self.original)

        for _ in range(amountOfInfinities):
            citya = random.randrange(0,self.original[0].size)
            cityb = random.randrange(0,self.original[0].size)
            while cityb == citya and self.original[citya][cityb] != np.inf:
                cityb = random.randrange(0, self.original[0].size)

            self.copy[citya][cityb] = np.inf


    def shortest_n(self,depth : int) -> np.array:
        cities = np.full(depth+1,-1)
        cities[0] = random.randrange(0,self.original[0].size)
        unusedIndexes = [i for i in range(self.original[0].size)]
        for i in range(depth):
            cityIdx = cities[i]
            unusedIndexes.remove(cityIdx)

            if self.closestCity[cityIdx] != -1:
                if not np.any(cities==self.closestCity[cityIdx]):
                    cities[i+1] = self.closestCity[cityIdx]
                    continue

            closest_city = -1
            closest_distance = np.inf
            for j in range(self.original[i].size):
                if (cityIdx != j and self.original[cities[i]][j] < closest_distance and not np.any(cities == j)):
                    closest_city = j
                    closest_distance = self.original[cityIdx][j]
            cities[i+1] = closest_city
            self.closestCity[cityIdx] = closest_city
        unusedIndexes.remove(cities[depth])
        unusedIndexes = np.array(unusedIndexes)
        np.random.shuffle(unusedIndexes)

        return np.concatenate((cities , unusedIndexes),axis=0)

    # Code inspired by https://github.com/pdrm83/py2opt/blob/master/py2opt/solver.py
    def two_opt(self, cities : np.array,heuristic : int, improvement_threshold=0.01) -> np.array:
        copy = np.copy(cities)
        best_distance = heuristic

        #self.best_route = self.initial_route
        #self.best_distance = self.calculate_path_dist(self.distance_matrix, self.best_route)
        improvement_factor = 1

        while improvement_factor > improvement_threshold:
            previous_best = best_distance
            for swap_first in range(1, cities.size - 2):
                mutated :bool  = True
                for swap_last in range(swap_first + 1, cities.size - 1):
                    if mutated:
                        partOnePath = copy[0:swap_first]
                        partOnePathEval = self.eval_subpath(partOnePath)
                        before_start = copy[swap_first - 1]
                        start = copy[swap_first]
                        mutated = False
                    end = copy[swap_last]
                    after_end = copy[swap_last + 1]
                    before = self.original[before_start][start] + self.original[end][after_end]
                    after = self.original[before_start][end] + self.original[start][after_end]
                    if after < before:
                        mutated = True
                        partTwoPath = copy[swap_last:-len(cities)+swap_first-1:-1]
                        partTwoPathEval = self.eval_subpath(partTwoPath)
                        partThreePath = copy[swap_last+1:len(cities)]
                        partThreePathEval = self.eval_subpath(partThreePath)
                        new_route = np.concatenate((partOnePath,partTwoPath,partThreePath))
                        new_distance = partOnePathEval + self.original[partOnePath[-1]][partTwoPath[0]] \
                                       + partTwoPathEval + self.original[partTwoPath[-1]][partThreePath[0]]\
                                       + partThreePathEval + self.original[partThreePath[-1]][partOnePath[0]]

                        copy = new_route
                        best_distance = new_distance

            improvement_factor = 1 - best_distance / previous_best
        return copy


    def two_opt_randomized(self, cities : np.array,heuristic : int, improvement_threshold=0.01) -> np.array:
        copy = np.copy(cities)
        best_distance = heuristic

        #self.best_route = self.initial_route
        #self.best_distance = self.calculate_path_dist(self.distance_matrix, self.best_route)
        improvement_factor = 1

        while improvement_factor > improvement_threshold:
            previous_best = best_distance
            swap_first = random.randrange(1,cities.size - 1)
            while swap_first != cities.size-2:
                mutated :bool  = True
                for swap_last in range(swap_first + 1, cities.size - 1):
                    if mutated:
                        partOnePath = copy[0:swap_first]
                        partOnePathEval = self.eval_subpath(partOnePath)
                        before_start = copy[swap_first - 1]
                        start = copy[swap_first]
                        mutated = False
                    end = copy[swap_last]
                    after_end = copy[swap_last + 1]
                    before = self.copy[before_start][start] + self.copy[end][after_end]
                    after = self.copy[before_start][end] + self.copy[start][after_end]
                    if after < before:
                        mutated = True
                        partTwoPath = copy[swap_last:-len(cities)+swap_first-1:-1]
                        partTwoPathEval = self.eval_subpath(partTwoPath)
                        partThreePath = copy[swap_last+1:len(cities)]
                        partThreePathEval = self.eval_subpath(partThreePath)
                        new_route = np.concatenate((partOnePath,partTwoPath,partThreePath))
                        new_distance = partOnePathEval + self.original[partOnePath[-1]][partTwoPath[0]] \
                                       + partTwoPathEval + self.original[partTwoPath[-1]][partThreePath[0]]\
                                       + partThreePathEval + self.original[partThreePath[-1]][partOnePath[0]]

                        copy = new_route
                        best_distance = new_distance
                swap_first += 1

            improvement_factor = 1 - best_distance / previous_best
        return copy
class Agent:
    cities: list[int]
    age: int
    score: int
    hasBeenModified: bool

    # Initiate Agent with age and list of cities in order of visitation.
    #
    def __init__(self, age, cities):
        self.cities = cities
        self.age = age
        self.score = 0
        self.hasBeenModified = True
        self.isBestAgent = False

    # Function that'll change the internal cities array depending on 2 parameters
    # 1. Amount of permutations: Represents the amount of random switches within the cities array
    # 2. Probability of Permutations: if random no. between [0;1] is smaller than prob. then switch will occur.
    def mutate(self, amount_of_permutations, probability_of_permutation):
        for i in range(amount_of_permutations):
            randomInt = random.random()
            # print("The random int generated is: ", randomInt , " And the prob. of permutation is: " , probability_of_permutation)
            if randomInt < probability_of_permutation:
                randa = random.randrange(len(self.cities))
                randb = random.randrange(len(self.cities))
                self.cities[randa], self.cities[randb] = self.cities[randb], self.cities[randa]
        self.hasBeenModified = True

    def mutateLocalSearch(self, distanceMatrix: np.array, depth: int):
        city = self.cities[0]

        for i in range(depth):
            closestcity = self.cities[i + 1]
            currentDistance = distanceMatrix[city][closestcity]
            for j in range(len(distanceMatrix[city])):
                if j == i:
                    continue

                if distanceMatrix[city][j] < currentDistance:
                    currentDistance = distanceMatrix[city][j]
                    closestcity = j

            idxOfClosestCity = np.where(self.cities == closestcity)
            self.cities[i + 1], self.cities[idxOfClosestCity[0]] = self.cities[idxOfClosestCity[0]], \
            self.cities[i + 1]

    def evaluateAgent(self, distanceMatrix: np.array) -> int:
        if not self.hasBeenModified:
            return self.score
        else:
            i = 0
            pathLength = 0
            cities: np.array = self.cities
            while i != len(cities) - 1:
                city_0 = cities[i]
                city_1 = cities[i + 1]

                if not isinf(distanceMatrix[city_0][city_1]):
                    pathLength += distanceMatrix[city_0][city_1]
                else:
                    pathLength *= 2
                i += 1
            pathLength += distanceMatrix[cities[i]][cities[0]]

            self.score = pathLength
            self.hasBeenModified = False
            return pathLength

    def mutateEliminateInfs(self, amount_of_permutations, probability_of_permutation,
                            pathManipulator: PathManipulator):
        if self.isBestAgent:
            return

        for i in range(amount_of_permutations):
            randomInt = random.random()
            # print("The random int generated is: ", randomInt , " And the prob. of permutation is: " , probability_of_permutation)
            if randomInt < probability_of_permutation:
                randa = random.randrange(len(self.cities))
                randb = random.randrange(len(self.cities))
                self.cities[randa], self.cities[randb] = self.cities[randb], self.cities[randa]

        self.cities = pathManipulator.remove_infinite_path_closest_city(self.cities)
        self.hasBeenModified = True

# Modify the class name to match your student number.
class r0736356:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        ##############################################################################
        # DEFINTION OF VARIABLES
        #
        amount_of_cities = distanceMatrix[0].size
        amount_of_permutations : int = amount_of_cities // 10
        probability_of_permutation : int = 0.3

        repeat_opt_counter = 0
        reset_counter = 3
        kTournamentSelectionSize = 5
        ##############################################################################
        # Vars Dependant on the population size:
        #############################################################################
        population_size = 0
        max_generations = 0
        if amount_of_cities <= 150:
            population_size = 200
            max_generations = 50
        elif amount_of_cities > 150 and amount_of_cities <= 450:
            population_size = 50
            max_generations = 31
        elif amount_of_cities > 450 and amount_of_cities <= 800 :
            population_size = 25
            max_generations = 20
        else:
            population_size = 20
            max_generations = 10

        current_generation = 0
        listOfBestSolutions = []

        pathManipulator = PathManipulator(distanceMatrix)
        agents = [Agent(0, pathManipulator.shortest_n(amount_of_cities//5)) for _ in range(population_size)]
        for agent in agents:
            pathManipulator.remove_infinite_path_closest_city(agent.cities)
            pathManipulator.hard_remove_infs(agent.cities)
        currentBestAgent : Agent = None


        while(current_generation != max_generations):
            current_generation += 1
            # First check if there needs to be a two_opt execution
            if population_size == 200 and current_generation % 4 == 0:
                for agent in agents:
                    agent.cities = pathManipulator.two_opt(agent.cities, agent.evaluateAgent(pathManipulator.copy))
                children = []
                for _ in range(population_size//2):
                    parent1 = r0736356.kTournamentSelection(list_of_agents=agents,k=kTournamentSelectionSize,pathManipulator=pathManipulator)
                    parent2 = r0736356.kTournamentSelection(list_of_agents=agents, k=kTournamentSelectionSize,pathManipulator=pathManipulator)
                    child = r0736356.orderedCrossoverVariant(parent1,parent2,current_generation)
                    children.append(child)
                agents = agents + children
            elif population_size < 200 and (current_generation % 4 == 0 or current_generation % 4 == 2):
                arr = []
                for _ in range(population_size // 10):
                    rand = random.randrange(0, population_size)
                    while rand in arr:
                       rand = random.randrange(0, population_size)
                    arr.append(arr)
                    currAgent = agents[rand]
                    currAgent.cities = pathManipulator.two_opt(currAgent.cities, currAgent.evaluateAgent(pathManipulator.copy))
                children = []
                for _ in range(population_size//2):
                    parent1 = r0736356.kTournamentSelection(list_of_agents=agents,k=kTournamentSelectionSize,pathManipulator=pathManipulator)
                    parent2 = r0736356.kTournamentSelection(list_of_agents=agents, k=kTournamentSelectionSize,pathManipulator=pathManipulator)
                    child = r0736356.orderedCrossoverVariant(parent1,parent2,current_generation)
                    children.append(child)
                agents = agents + children
            elif (current_generation % 4) == 1:
                children = []
                for _ in range(population_size // 4):
                    parent1 = r0736356.kTournamentSelection(list_of_agents=agents, k=kTournamentSelectionSize,
                                                            pathManipulator=pathManipulator)
                    parent2 = r0736356.kTournamentSelection(list_of_agents=agents, k=kTournamentSelectionSize,
                                                            pathManipulator=pathManipulator)
                    child1,child2 = r0736356.CX2_Crossover(parent1.cities, parent2.cities, current_generation)
                    children.append(child1)
                    children.append(child2)
                agents = agents + children
            else:
                children = []
                for _ in range(population_size//2):
                    parent1 = r0736356.kTournamentSelection(list_of_agents=agents,k=kTournamentSelectionSize,pathManipulator=pathManipulator)
                    parent2 = r0736356.kTournamentSelection(list_of_agents=agents, k=kTournamentSelectionSize,pathManipulator=pathManipulator)
                    child = r0736356.orderedCrossoverVariant(parent1,parent2,current_generation)
                    children.append(child)
                agents = agents + children

            ## Agents have all had crossover,
            # Now onto mutation
            r0736356.mutate(list_of_agents=agents,pathManipulator=pathManipulator,amount_of_perms=amount_of_permutations,prob_of_perm=probability_of_permutation)
            agents = r0736356.eliminate(list_of_agents=agents,k=kTournamentSelectionSize,pathManipulator=pathManipulator,population_size=population_size)

            ## Calculation of all the objective values
            fitness_scores = [agent.evaluateAgent(pathManipulator.original) for agent in agents]
            min_score = min(fitness_scores)
            best_agent :Agent= agents[fitness_scores.index(min_score)]
            meanObjective = sum(fitness_scores)/len(fitness_scores)


            if not best_agent.isBestAgent and currentBestAgent is None:
                best_agent.isBestAgent = True
                currentBestAgent = best_agent
            elif not best_agent.isBestAgent and currentBestAgent.score > best_agent.score:
                currentBestAgent.isBestAgent = False
                currentBestAgent = best_agent
                best_agent.isBestAgent = True
            elif not best_agent.isBestAgent and currentBestAgent.score < best_agent.score:
                pass

            bestObjective = currentBestAgent.score
            bestSolution = currentBestAgent.cities
            bestSolution = np.roll(bestSolution, -np.where(bestSolution == 0)[0])

            prev_bestObjective = 0
            if current_generation >= 2:
                prev_bestObjective = listOfBestSolutions[-1]

            if bestObjective == prev_bestObjective:
                repeat_opt_counter += 1
            else:
                repeat_opt_counter = 0
                reset_counter = 0

            if repeat_opt_counter >= 3 and reset_counter <= 0 and current_generation < max_generations - 10:
                probability_of_permutation = 0.7
                amount_of_permutations = amount_of_cities//5
                reset_counter = 2
                repeat_opt_counter = 0
                pathManipulator.introduce_infinities(amount_of_cities//5)
                for agent in agents:
                    agent.hasBeenModified = True
            elif reset_counter > 0:
                reset_counter = reset_counter - 1
            elif reset_counter == 0:
                probability_of_permutation = 0.3
                amount_of_permutations = amount_of_cities // 10
                pathManipulator.introduce_infinities(0)

            listOfBestSolutions.append(bestObjective)


            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

        # Your code here.
        return 0


    @staticmethod
    def mutate(list_of_agents : [Agent],amount_of_perms : int, prob_of_perm : int,pathManipulator : PathManipulator):
        for agent in list_of_agents:
            if agent.isBestAgent:
                continue
            agent.mutate(amount_of_perms,prob_of_perm)
            agent.cities = pathManipulator.hard_remove_infs(agent.cities)

    @staticmethod
    def kTournamentSelection(list_of_agents : [Agent],k : int,pathManipulator: PathManipulator):
        agent_sample = random.sample(list_of_agents,k)
        fitness_scores = [agent.evaluateAgent(pathManipulator.copy) for agent in agent_sample]
        return list_of_agents[list_of_agents.index(agent_sample[fitness_scores.index(min(fitness_scores))])]

    @staticmethod
    def kTournamentElimination(list_of_agents: [Agent], k: int, pathManipulator: PathManipulator):
        agent_sample = random.sample(list_of_agents, k)
        fitness_scores = [agent.evaluateAgent(pathManipulator.copy) for agent in agent_sample]
        return list_of_agents[list_of_agents.index(agent_sample[fitness_scores.index(max(fitness_scores))])]

    @staticmethod
    def eliminate(list_of_agents: [Agent],population_size:int, k: int, pathManipulator: PathManipulator):
        while len(list_of_agents) > population_size:
            list_of_agents.remove(r0736356.kTournamentElimination(list_of_agents, k, pathManipulator))
        return list_of_agents

    @staticmethod
    def orderedCrossoverVariant(agent1:Agent,agent2:Agent,generation:int):
        new_cities = agent1.cities
        p2_cities = agent2.cities
        subset_start = random.randrange(0, len(new_cities) - 1)
        subset_length = random.randrange(1, len(new_cities) - subset_start)

        subset = new_cities[subset_start:subset_start + subset_length]
        new_positions = random.sample(range(0, len(new_cities)), subset_length)
        new_positions.sort()

        subset_index = 0

        new_cities = [-1] * len(p2_cities)

        for i in range(len(new_cities)):
            if i in new_positions:
                new_cities[i] = subset[subset_index]
                subset_index += 1
            else:
                for city in agent2.cities:
                    if city not in subset and city not in new_cities:
                        new_cities[i] = city
                        break

        return Agent(generation, np.array(new_cities))

    @staticmethod
    def CX2_Crossover(cities1: np.array,cities2 : np.array,generation:int) -> tuple[Agent,Agent]:
        # Initialize the offspring as copies of the parents
        offspring1 = np.copy(cities1)
        offspring2 = np.copy(cities2)

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
                k = np.where(cities1 == cities2[j])
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


        return Agent(generation, offspring1), Agent(generation, offspring2)






