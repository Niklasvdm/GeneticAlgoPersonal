import numpy as np
import matplotlib.pyplot as plt


file = open("/home/Universiteit/GeneticAlgorithms/GeneticAlgoPersonal/PythonCode/Averages.csv")
distanceMatrix = file.read()
file.close()

print(distanceMatrix)
actualArray = []
str = ""
point = False
for i in distanceMatrix:
    if i == ",":
        actualArray.append(int(str))
        str = ""
        point = False
    elif i == "[" or i =="]":
        pass
    elif i == '.':
        point = True
    elif not point:
        str = str + i


print(actualArray)
dict_avg = {}
for avg in actualArray:
    avg = round(avg / 10000) * 10000
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
