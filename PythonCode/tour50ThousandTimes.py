#from TravellingSalesman import r0736356
import os
from Population import Population
import numpy as np
import matplotlib.pyplot as plt

file = open("/home/Universiteit/GeneticAlgorithms/tours/tour50.csv")
distanceMatrix = np.loadtxt(file, delimiter=",")
file.close()
cities = list(range(distanceMatrix[0].size))

#
Thousand_min_scores = []
Thousand_avg_scores = []

for _ in range(1000):
    # Your code here.
    population = Population(200, distanceMatrix, cities)
    counter = 0
    resetCounter = 3

    min_score, avg_score = 0, 0
    lowest_score, lowest_avg = np.inf,np.inf
    bestSolution = []
    x_val = 0

    min_scores = []
    avg_scores = []
    x = []
    MAX_RUNS = 50
    runs = 0
    # TODO: THIS LOOP NEEDS TO STOP
    haschanged = False
    while True:#avg_score >= min_score * 1.005 or avg_score == 0:
        runs+=1
        population.age = runs
        if runs>MAX_RUNS:
            print("I've reached ", MAX_RUNS, " runs")
            break


        if runs == 0:
            #population.applyTwoOpt()
            #population.applyTwoOptRandomized()
            pass
        elif runs % 4 == 0 :#or runs % 4 == 2:
            #population.applyTwoOptRandomized()
            population.applyTwoOpt()
            #population.applyTwoOptRandomSubset()
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
        elif resetCounter > 0:
            resetCounter = resetCounter - 1
        elif resetCounter == 0:
            if haschanged:
                print("let's reset")
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

            min_scores.append(min_score)
            avg_scores.append(avg_score)

            x_val += 10
            x.append(x_val)


    Thousand_min_scores.append(lowest_score)
    Thousand_avg_scores.append(avg_score)
dict_lowest = {}
dict_avg = {}
for lowest in Thousand_min_scores:
    lowest = round(lowest/1000)*1000
    if lowest in dict_lowest:
        dict_lowest[lowest] += 1
    else:
        dict_lowest[lowest] = 1
for avg in Thousand_avg_scores:
    avg = round(avg / 1000) * 1000
    if avg in dict_avg:
        dict_avg[avg] += 1
    else:
        dict_avg[avg] = 1


dict_avg_accum = [(key,value) for key,value in dict_avg.items()]
dict_avg_accum = sorted(dict_avg_accum)
y_pos = np.arange(len(dict_avg_accum))
plt.bar(y_pos,[value for (key,value) in dict_avg_accum])
plt.xticks(y_pos,[key for (key,value) in dict_avg_accum])
plt.ylabel("Amount")
plt.xlabel("Average")
plt.show()

dict_lowest_accum = [(key,value) for key,value in dict_lowest.items()]
dict_lowest_accum = sorted(dict_lowest_accum)
y_pos = np.arange(len(dict_lowest_accum))
plt.bar(y_pos,[value for (key,value) in dict_lowest_accum])
plt.xticks(y_pos,[key for (key,value) in dict_lowest_accum])
plt.ylabel("Amount")
plt.xlabel("Lowest")
plt.show()

print("Standard deviation of Averages is: ", np.std(Thousand_avg_scores))
print("Standard deviation of Lowest is: ", np.std(Thousand_min_scores))

print("Mean of Averages is: " , sum(Thousand_avg_scores)/len(Thousand_avg_scores))
print("Mean of Lowest is: " , sum(Thousand_min_scores)/len(Thousand_min_scores))

with open("Averages.csv",'w') as f:
    f.write(str(Thousand_avg_scores))
f.close()
with open("Minima.csv",'w') as f:
    f.write(str(Thousand_min_scores))
f.close()












