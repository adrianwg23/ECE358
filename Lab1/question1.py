import datetime
import random
import math
import matplotlib.pyplot as plt
import numpy

LAMBDA = 75

def exponential_random_variable():
    random.seed(datetime.datetime.now())
    trials = numpy.array([])
    for _ in range(1000):
        U = random.random()
        x = -(1/LAMBDA)*math.log(1-U)
        trials = numpy.append(trials, x)
    
    E = 1/LAMBDA
    E_array = numpy.mean(trials)
    print("Expected value for Exponential Random Variable: {}".format(E))
    print("Expected value for generated array: {}".format(E_array))

    var = 1/LAMBDA**2
    var_array = numpy.var(trials)
    print("Variance for Exponential Random Variable: {}".format(var))
    print("Variance for generated array: {}".format(var_array))
 
if __name__ == "__main__":
    exponential_random_variable()