import datetime
import random
import math
import matplotlib.pyplot as plt
import numpy

from collections import deque


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


def plot_graphs(En, Pidles):
    rho = [.25,.35,.45,.55,.65,.75,.85,.95]
    # Plot En vs rho
    plt.plot(rho, En)
    plt.ylabel('Average number in system (E[N])')
    plt.xlabel('Traffic intensity (rho)')
    plt.show()

    # Plot Pidles vs rho
    plt.plot(rho, Pidles)
    plt.ylabel('Pidle')
    plt.xlabel('Traffic intensity (rho)')
    plt.show()


class EventQueue:
    def __init__(self, rho, L, C, simulation_time) -> None:
        self.rho = rho
        self.L = L
        self.C = C
        self.simulation_time = simulation_time
        self.Na = 0
        self.Nd = 0
        self.No = 0

        self.queue = deque()
        self.num_packets_in_buffer = []
        self.idle_time = 0
        self.Pidle = 0
        self.Ploss = 0
        self.average_num_of_packets = 0

    def run_simulation(self):
        last_departure_time = 0
        last_oberserver_time = 0
        idle_in_progress = False
        while self.queue:
            event, current_time = self.queue.popleft()
            if event != "observer" and idle_in_progress:
                self.idle_time += (last_oberserver_time-last_departure_time)
                idle_in_progress = False
            if event == "arrival":
                self.Na -= 1
            elif event == "departure":
                self.Nd -= 1
                last_departure_time = current_time
            else:
                self.No -= 1
                num_of_packets_in_buffer = self.Nd - self.Na
                self.num_packets_in_buffer.append(num_of_packets_in_buffer)
                if num_of_packets_in_buffer == 0:
                    last_oberserver_time = current_time
                    idle_in_progress = True

        self.average_num_of_packets = sum(self.num_packets_in_buffer) / len(self.num_packets_in_buffer)
        self.Pidle = self.idle_time / self.simulation_time

    def create_queue(self):
        arrivals = []
        departures = []
        observers = []

        current_time = 0
        while current_time < self.simulation_time:
            arrival_rate = compute_arrival_rate(self.rho, self.L, self.C)
            arrival_time = exponential_random_variable(arrival_rate) + current_time
            arrivals.append(('arrival', arrival_time))
            
            packet_length = exponential_random_variable(1/self.L)
            service_time = compute_service_time(packet_length, self.C)

            if len(departures) == 0 or departures[-1][1] < arrival_time:
                departure_time = arrival_time + service_time
            else:
                departure_time = departures[-1][1] + service_time

            departures.append(('departure', departure_time))

            current_time = arrival_time

        current_time = 0
        while current_time < self.simulation_time:
            observer_time = exponential_random_variable(5*arrival_rate) + current_time
            observers.append(('observer', observer_time))

            current_time = observer_time

        self.Na = len(arrivals)
        self.Nd = len(departures)
        self.No = len(observers)
        all_events = arrivals + departures + observers
        all_events.sort(key=lambda x: x[1])
        self.queue = deque(all_events)

if __name__ == "__main__":
    En = []
    Pidles = [] 
    for i in range(25, 96, 10):
        Queue = EventQueue(i/100, 2000, 1000000, 1000)
        Queue.create_queue()
        Queue.run_simulation()
        En.append(Queue.average_num_of_packets)
        Pidles.append(Queue.Pidle)
    print(En)
    print(Pidles)
    plot_graphs(En, Pidles)