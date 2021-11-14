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

def compute_transmission_delay(L, R):
    return L/R

def compute_propagation_delay(D, S):
    return D/S


class Node:
    def __init__(self, A):
        self.queue = []
        self.backoff_counter = 0
        A = A # average packet arrival rate

    def create_queue(self):
        # create list of arrival events using exponential_random_variable for each node 
        pass


class PersisentCSMACD:
    def __init__(self, N, R, L, D, S, T):
        self.N = N # number of nodes
        self.R = R # speed of the LAN
        self.L = L # packet length
        self.D = D # distance between adjacent nodes on bus
        self.S = S # propagation speed
        self.T = T # simulation time
        self.nodes = []

    def create_nodes(self):
        # create N nodes
        # for each node, call its method create_queue
        pass

    def run_simulation(self):
        # check which arrival time is the smallest among the nodes and get node index
        # process event at node index
        pass
    
    def calculate_exp_backoff_time(self, node):
        # do bit-time calculate (should be 512 microseconds)
        # returns random int number between (0, 2^i-1) * 512 bit-time
        pass

    def process_event(self, node_index, curr_time):
        # var: collision_detected = False
        # var: curr_time = curr_time
        # var: max distance of collision
        
        # loop go right [node_index to end of  node array]
            # curr_time += propagation_delay*(index-node_index)
            # check if it collides, if so
                # T(receiver) = T(sender) + Tprop*(index-node_index) + Ttrans + Texp_backoff(receiver)       ||| bubble this time up to all events in queue for that node
                # update max distance of collision
                # set collision_detected to True

        # loop go left[node_index to beginning of  node array]
            # curr_time += propagation_delay*(index-node_index)
            # check if it collides, if so
                # T(receiver) = T(sender) + Tprop*(index-node_index) + Ttrans + Texp_backoff(receiver)       ||| bubble this time up to all events in queue for that node
                # update max distance of collision
                # set collision_detected to True
        
        # check collision flag, if True,
            # update T(sender) = T(sender) + Tprop*(max_distance) + Texp_backoff(sender)     ||| bubble this time up to all events in queue for that node
        # if False,
            # loop through each node again and find which events need a +Ttrans by following inequality thing
            # if not busy, we good
            # if busy which implies T(curr_index) < T(sender + Tprop*distance + Ttrans
                # bubble up the T(sender + Tprop*distance + Ttrans to all of events in the current node where that number is less than the event time
        pass


class NonPersisentCSMACD:
    pass 

if __name__ == "__main__":
    print("Done.")
