import datetime
import random
import math
import matplotlib.pyplot as plt
import numpy

from collections import deque


def exponential_random_variable_question1():
    """
    Generates 1000 expontential random variables and computes its expected value and variance for Question 1.
    """
    LAMBDA = 75
    random.seed(float(datetime.datetime.now().timestamp()))
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


def question3_plot_graphs(rhos, En, Pidle):
    plt.figure(1)
    # Plot En vs rho
    plt.plot(rhos, En)
    plt.title("En vs. rho")
    plt.ylabel('Average number in system (E[N])')
    plt.xlabel('Traffic intensity (rho)')
    plt.show()

    plt.figure(2)
    # Plot Pidle vs rho
    plt.plot(rhos, Pidle)
    plt.title("Pidle vs. rho")
    plt.ylabel('Pidle')
    plt.xlabel('Traffic intensity (rho)')
    plt.show()


def question6_plot_graphs(rhos, Ens, Plosses):
    K_values = [10, 25, 50]
    i = 0
    plt.figure(3)
    for En in Ens:
        # Plot En vs rho
        plt.plot(rhos, En, label="K={}".format(K_values[i]))
        plt.title("En vs. rho for each K")
        plt.legend(loc="upper left")
        plt.ylabel('Average number in system (E[N])')
        plt.xlabel('Traffic intensity (rho)')
        i += 1
    plt.show()

    i = 0
    plt.figure(4)
    for Ploss in Plosses:
        # Plot Ploss vs rho
        plt.plot(rhos, Ploss, label="K={}".format(K_values[i]))
        plt.title("Ploss vs. rho for each K")
        plt.legend(loc="upper left")
        plt.ylabel('Ploss')
        plt.xlabel('Traffic intensity (rho)')
        i += 1
    plt.show()


class InfiniteEventQueue:
    """A class used to represent a M/M/1 network queue"""

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
        """
        Processes all Arrival, Departure, and Observer events from queue and computes the average 
        number of packets and idle time for the entire simulation time.
        """
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
        """
        Creates Arrival, Departure, and Observer events based on rho, L, and C values and appends them 
        to the M/M/1 queue.
        """
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


class FiniteEventQueue:
    """A class used to represent a M/M/1/K network queue"""

    def __init__(self, rho, L, C, simulation_time, K) -> None:
        self.rho = rho
        self.L = L
        self.C = C
        self.simulation_time = simulation_time
        self.K = K
        self.Na = 0
        self.No = 0
        self.num_of_packet_loss = 0

        self.queue = deque()
        self.buffer = deque()
        self.num_packets_in_buffer = []
        self.idle_time = 0
        self.Pidle = 0
        self.Ploss = 0
        self.average_num_of_packets = 0

    def run_simulation(self):
        """
        Process all Arrival and Observer events in the queue. Departure events are generated on the fly when a
        Arrival events gets added to the buffer. A counter for packet loss is tracked when each packet attempts
        to enter the buffer. The average number of packets, Pidle, and Ploss are computed at the end.
        """
        last_departure_time = 0
        last_oberserver_time = 0
        idle_in_progress = False
        while self.queue:
            event, current_time = self.queue.popleft()
            while self.buffer and current_time > self.buffer[0]:
                last_departure_time = self.buffer.popleft()
            if event == "arrival":
                if idle_in_progress:
                    self.idle_time += (last_oberserver_time-last_departure_time)
                    idle_in_progress = False
                if len(self.buffer) >= self.K:
                    self.num_of_packet_loss += 1
                else:
                    # Generate departure event
                    packet_length = exponential_random_variable(1/self.L)
                    service_time = compute_service_time(packet_length, self.C)
                    if len(self.buffer) == 0:
                        corresponding_departure_time = current_time + service_time
                    else:
                        corresponding_departure_time = self.buffer[-1] + service_time
                    self.buffer.append(corresponding_departure_time)
            else:
                self.num_packets_in_buffer.append(len(self.buffer))
                if not self.buffer:
                    last_oberserver_time = current_time
                    idle_in_progress = True

        self.average_num_of_packets = sum(self.num_packets_in_buffer) / len(self.num_packets_in_buffer)
        self.Pidle = self.idle_time / self.simulation_time
        self.Ploss = self.num_of_packet_loss / self.Na

    def create_queue(self):
        """
        Creates a queue of arrival and observer events. These are appended into the queue in 
        chronological order.
        """
        arrivals = []
        observers = []

        current_time = 0
        while current_time < self.simulation_time:
            arrival_rate = compute_arrival_rate(self.rho, self.L, self.C)
            arrival_time = exponential_random_variable(arrival_rate) + current_time
            arrivals.append(('arrival', arrival_time))

            current_time = arrival_time

        current_time = 0
        while current_time < self.simulation_time:
            observer_time = exponential_random_variable(5*arrival_rate) + current_time
            observers.append(('observer', observer_time))

            current_time = observer_time

        self.Na = len(arrivals)
        self.No = len(observers)
        all_events = arrivals + observers
        all_events.sort(key=lambda x: x[1])
        self.queue = deque(all_events)


if __name__ == "__main__":
    # Question 1
    print("Simulating Question 1...")
    exponential_random_variable_question1()
    print("")

    # Question 3
    print("Simulating Question 3...")
    question3_En = []
    question3_Pidles = []
    question3_rhos = [.25, .35, .45, .55, .65, .75, .85, .95]
    for rho in question3_rhos:
        Queue = InfiniteEventQueue(rho, 2000, 1000000, 1000)
        Queue.create_queue()
        Queue.run_simulation()
        question3_En.append(Queue.average_num_of_packets)
        question3_Pidles.append(Queue.Pidle)
        print("Infinite Queue - Finished simulating rho={}".format(rho))
    print("")
    
    # Question4
    print("Simulating Question 4...")
    question4_En = []
    question4_Pidles = []
    Queue = InfiniteEventQueue(1.2, 2000, 1000000, 1000)
    Queue.create_queue()
    Queue.run_simulation()
    print("Question 4: En={} for rho=1.2".format(Queue.average_num_of_packets))
    print("Question 4: Pidle ={} for rho=1.2".format(Queue.Pidle))
    print("")

    # Question6
    print("Simulating Question 6...")
    Ks = [10, 25, 50]
    question6_rhos = [.5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    question6_En = []
    question6_Plosses = []
    for K in Ks:
        En = []
        Plosses = []
        for rho in question6_rhos:
            Queue = FiniteEventQueue(rho, 2000, 1000000, 1000, K)
            Queue.create_queue()
            Queue.run_simulation()
            En.append(Queue.average_num_of_packets)
            Plosses.append(Queue.Ploss)
            print("Finite Queue - Finished simulating rho={}".format(rho))
        question6_En.append(En)
        question6_Plosses.append(Plosses)
    print("")
    
    # Plot all graphs
    question3_plot_graphs(question3_rhos, question3_En, question3_Pidles)
    question6_plot_graphs(question6_rhos, question6_En, question6_Plosses)
    print("Done.")
