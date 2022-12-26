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
import Reporter
import numpy as np
from Population import Population
import matplotlib.pyplot as plt


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
        population = Population(200, distanceMatrix, cities)
        counter = 0
        resetCounter = 3
        yourConvergenceTestsHere = True
        min_score, avg_score = 0, 0
        bestSolution = []
        x_val = 0

        min_scores = []
        avg_scores = []
        x = []

        runs = 0
        while avg_score >= min_score * 1.005 or avg_score == 0:
            runs+=1
            if runs>500:
                print("I've reached 50 runs")
                break

            print(min_score, avg_score)
            # Your code here.
            population.crossover()
            population.mutate()
            population.eliminate()
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            min_score, avg_score, bestSolution = population.getObjectiveValues()
            if len(min_scores) > 10 and min_score == min_scores[-1]:
                counter += 1
            else:
                counter = 0

            if counter >= 3 and resetCounter <= 0:
                population.probability_of_permutation = 0.7
                population.amount_of_permutations = len(cities)//5
                resetCounter = 5
                population.pathManipulator.introduce_infinities(len(cities)//1000)
            else:
                resetCounter = resetCounter - 1
                population.probability_of_permutation = 0.3
                population.amount_of_permutations = len(cities)//10


            min_scores.append(min_score)
            avg_scores.append(avg_score)

            x_val += 10
            x.append(x_val)

            timeLeft = self.reporter.report(avg_score, min_score, np.array(bestSolution))
            if timeLeft < 0:
                print("NO MORE TIME LEFT,STOPPING EXECUTION")
                break

        # Your code here.
        print(min_score, avg_score, bestSolution)
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(x, min_scores, color="red")
        plt.plot(x, avg_scores, color="blue")
        plt.show()
        return 0


if __name__ == "__main__":
    tsproblem = r0736356()
    tsproblem.optimize("/home/Universiteit/GeneticAlgorithms/tours/tour250.csv")
