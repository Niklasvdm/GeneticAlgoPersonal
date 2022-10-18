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

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		# Your code here.
		yourConvergenceTestsHere = True
		while( yourConvergenceTestsHere ):
			meanObjective = 0.0
			bestObjective = 0.0
			bestSolution = np.array([1,2,3,4,5])

			# Your code here.

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
