import datetime
import random
import math
import matplotlib.pyplot as plt
import numpy

from collections import deque


def exponential_random_variable(lambda_value):
    """
    Generates a exponential random variable with lambda input
    """
    random.seed(float(datetime.datetime.now().timestamp()))
    U = random.random()
    x = -(1/lambda_value)*math.log(1-U)
    
    return x


def compute_arrival_rate(rho, L, C):
    return rho * compute_service_rate(L, C)


def compute_service_rate(L, C):
    return C/L


def compute_service_time(L, C):
    return L/C



if __name__ == "__main__":
    print("Done.")
