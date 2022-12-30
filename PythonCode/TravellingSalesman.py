# Cities = [x1,x2,x3,...,xn]
# Agents = [x4, x5 , ... ] = Order of visitations
# Variation operator = Execute (random) partial permutation on the agent visitiation list
# {x1 : { x2 : _ , x3 : _} } Weighted directed graph representation is a nested dictionary. Space complexity not so good, time complexity is nice.
# ~~~Alternatively: use matrix representation of NxN Matrix with cities
#
# Selection Operator: Minimal length possibly combined with Age value.
# K-Tournament selection.
#
# Recombination: If parts of visitation array (= agent array) are similar, keep similarities and
# 				introduce variation into array subset that's different.
#
# Conversion Test: Example: If improvement over X generations isn't more than 10% of total travel length
#
import os

import Reporter
import numpy as np
from Population import Population
import matplotlib.pyplot as plt
from Agent import Agent
from TestUtilFunctions import has_duplicates

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

        cities = list(range(distanceMatrix[0].size))

        # Your code here.
        population = Population(25, distanceMatrix, cities)
        counter = 0
        resetCounter = 3
        yourConvergenceTestsHere = True
        min_score, avg_score = 0, 0
        lowest_score, lowest_avg = np.inf,np.inf
        bestSolution = []
        x_val = 0

        MAX_RUNS = 15

        min_scores = []
        avg_scores = []
        best_path : np.array
        x = []

        runs = 0
        haschanged = False
        while True:#avg_score >= min_score * 1.005 or avg_score == 0:
            population.age = runs
            if runs>MAX_RUNS:
                print("I've reached ", MAX_RUNS, " runs")
                break


            # if runs % 5 == 0:
            #     os.environ["MUTATE"] = "RANDOMOPTIMIZED"
            # else:
            #     os.environ["MUTATE"] = "RANDOMSWAPELIMINFS"

            if runs == 0:

                #population.applyTwoOpt()
                #population.applyTwoOptRandomized()
                pass
            elif runs % 4 == 0 or runs % 4 == 2:
                #population.applyTwoOptRandomized()
                #population.applyTwoOpt()
                population.applyTwoOptRandomSubset()
                #os.environ["CROSSOVER"] = "OPTEXCLUSIVITY"
            elif runs % 4 == 1:# or runs % 4 == 3:
                os.environ["CROSSOVER"] = "CYCLE2"
            else:
                os.environ["CROSSOVER"] = "DEFAULT"





            print(runs , min_score, avg_score)
            # Your code here.
            #population.applyTwoOpt()

            population.crossover()
            population.mutate()
            population.eliminate()
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            min_score, avg_score, bestSolution = population.getObjectiveValues()
            bestSolution = np.roll(bestSolution,-np.where(bestSolution == 0)[0])
            if len(min_scores) > 5 and min_score == min_scores[-1]:
                counter += 1
            else:
                counter = 0
                resetCounter = 0

            if counter >= 3 and resetCounter <= 0 and runs < MAX_RUNS - 10:
                print("Eyy let's make this crazy")
                haschanged = True
                population.probability_of_permutation = 0.7
                population.amount_of_permutations = len(cities)//5
                resetCounter = 2
                counter = 0
                population.pathManipulator.introduce_infinities(len(cities)//5)
                population.resetAgentScores()
                #os.environ["ELIMINATION"] = "FITNESSSHARING"
                #os.environ["CROSSOVER"] = "DEFAULT"
                # for _ in range(population.size // 2):
                #     population.agents.append(Agent(0,population.pathManipulator.shortest_n(25)))
            elif resetCounter > 0:
                resetCounter = resetCounter - 1

            elif resetCounter == 0:
                if haschanged:
                    print("let's reset")
                    haschanged = not haschanged
                resetCounter = resetCounter - 1
                population.probability_of_permutation = 0.3
                population.amount_of_permutations = len(cities)//10
                #TODO: TRY REMOVING/PLAYING WITH THIS
                population.pathManipulator.introduce_infinities(0)
                #os.environ["CROSSOVER"] = "ORDERED"

                #os.environ["ELIMINATION"] = "DEFAULT"

            if (min_score < lowest_score):
                lowest_score = min_score
                lowest_avg = avg_score
                best_path = bestSolution

            min_scores.append(min_score)
            avg_scores.append(avg_score)

            x_val += 10
            x.append(x_val)

            timeLeft = self.reporter.report(avg_score, min_score, np.array(bestSolution))
            if timeLeft < 0:
                print("NO MORE TIME LEFT,STOPPING EXECUTION")
                break
            runs+=1
        # Your code here.

        print("\n Lowest values are:", lowest_score,lowest_avg , "\n")
        print("City Array w/ best scores: " , best_path)
        print(min_score, avg_score, bestSolution)
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(x, min_scores, color="red")
        plt.plot(x, avg_scores, color="blue")
        plt.show()
        performance = set(agents.age for agents in population.agents)
        ## Make list with all ages -> Count occurences bound in map -> Output ones where != 0
        ages = [agent.age for agent in population.agents]
        scores = [agent.score for agent in population.agents]
        print(ages)
        print(scores)
        dict = {}
        for age in ages:
            if age in dict.keys():
                dict[age] += 1
            else:
                dict[age] = 1
        print(dict)
        dict_accum = [(key,value) for key,value in dict.items()]
        print(dict_accum)
        y_pos = np.arange(len(dict_accum))
        plt.bar(y_pos,[value for (key,value) in dict_accum])
        plt.xticks(y_pos,[key for (key,value) in dict_accum])
        plt.ylabel("Amount")
        plt.xlabel("Age of creation")
        plt.show()

        print("Mean of Averages is: ", sum(ages) / len(ages))
        print("Mean of Lowest is: ", sum(scores) / len(scores))

        return 0


if __name__ == "__main__":
    tsproblem = r0736356()
    tsproblem.optimize("/home/Universiteit/GeneticAlgorithms/tours/tour1000.csv")
