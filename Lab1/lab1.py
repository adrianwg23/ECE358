import datetime
import random
import math
import matplotlib.pyplot as plt
import numpy

LAMBDA = 75

def exponential_random_variable_question1():
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


def exponential_random_variable(lambda_value):
    random.seed(datetime.datetime.now())
    U = random.random()
    x = -(1/lambda_value)*math.log(1-U)
    
    return x


def compute_arrival_rate(rho, L, C):
    return rho * compute_service_rate(L, C)


def compute_service_rate(L, C):
    return C/L


def compute_service_time(L, C):
    return L/C


class EventQueue:
    def __init__(self) -> None:
        self.queue = []

    def create_queue(self, rho, L, C, simulation_time):
        arrivals = []
        departures = []
        observers = []

        current_time = 0
        while current_time < simulation_time:
            arrival_rate = compute_arrival_rate(rho, L, C)
            arrival_time = exponential_random_variable(arrival_rate) + current_time
            arrivals.append(('arrival', arrival_time))
            
            packet_length = exponential_random_variable(1/L)
            service_time = compute_service_time(packet_length, C)

            if len(departures) == 0 or departures[-1][1] < arrival_time:
                departure_time = arrival_time + service_time
            else:
                departure_time = departures[-1][1] + service_time

            departures.append(('departure', departure_time))
            
            current_time = arrival_time

        current_time = 0
        while current_time < simulation_time:
            observer_time = exponential_random_variable(5*arrival_rate) + current_time
            observers.append(('observer', observer_time))

            current_time = observer_time

        self.queue = arrivals + departures + observers
        self.queue.sort(key=lambda x: x[1])

        print(len(arrivals))
        print(len(departures))
        print(len(observers))


if __name__ == "__main__":
    solution = EventQueue()
    solution.create_queue(0.25, 2000, 1000000, 10)
    