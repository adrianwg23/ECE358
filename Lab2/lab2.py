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
    def __init__(self, A, T):
        self.queue = deque()
        self.backoff_counter = 0
        self.max_backoff = 10
        self.A = A # average packet arrival rate
        self.T = T # simulation time
        self.completed = False

    def create_queue(self):
        current_time = 0
        # create events up to simulation time T
        while True:
            arrival_time = exponential_random_variable(self.A) + current_time
            if arrival_time > self.T:
                break

            self.queue.append(arrival_time)
            current_time = arrival_time

    def should_drop_packet(self):
        return self.backoff_counter > self.max_backoff

    def drop_packet(self):
        self.queue.popleft()
        self.backoff_counter = 0


class PersisentCSMACD:
    def __init__(self, N, A, T):
        self.N = N # number of nodes
        self.A = A # average packet arrival rate
        self.T = T # simulation time

        # constants
        self.R = 10**6 # speed of the LAN
        self.L = 1500 # packet length
        self.D = 10 # distance between adjacent nodes on bus
        self.S = (2/3) * (3 * (10**8)) # propagation speed

        self.dropped_packets = 0
        self.completed_nodes = 0
        self.total_packets = 0
        self.nodes = []

    def create_nodes(self):
        for _ in range(self.N):
            node = Node(self.A, self.T)
            node.create_queue()
            self.total_packets += len(node.queue)
            self.nodes.append(node)

    def run_simulation(self):
        self.create_nodes()
        # check which arrival time is the smallest among the nodes and get node index
        while self.completed_nodes < self.N:
            min_node_index = 0
            min_timestamp = float('inf')
            for i, node in enumerate(self.nodes):
                if node.queue and node.queue[0] < min_timestamp:
                    min_timestamp = node.queue[0]
                    min_node_index = i

            # process event at node index
            self.process_event(min_node_index, min_timestamp)
    
    def calculate_exp_backoff_time(self, node):
        Tp = 512 / self.R
        return random.randint(0, (2**node.backoff_counter) - 1) * Tp

    def process_event(self, sender_index, sender_frame_time):
        collision_detected = False
        max_distance = float('-inf')

        self.check_collision(sender_index, sender_frame_time, collision_detected, max_distance, "right")
        self.check_collision(sender_index, sender_frame_time, collision_detected, max_distance, "left")
        
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

    def check_collision(self, sender_index, sender_frame_time, collision_detected, max_distance, direction):
        curr_index = sender_index
        curr_index = self.get_next_index(curr_index, direction)

        while True:
            curr_node = self.nodes[curr_index]
            head_frame_time = curr_node.queue[0]
            distance_to_sender = abs(curr_index - sender_index)

            # if the current node has no more events to offer in its queue, go next
            if not curr_node.queue:
                curr_index = self.get_next_index(curr_index, direction)
                continue

            received_time = sender_frame_time + (compute_propagation_delay(self.D, self.S) * (distance_to_sender))

            # collision occurs
            if head_frame_time <= received_time:
                curr_node.backoff_counter += 1

                if curr_node.should_drop_packet():
                    curr_node.drop_packet()
                    self.dropped_packets += 1
                else:
                    new_receiver_frame_time = received_time + compute_transmission_delay(self.L, self.R) + self.calculate_exp_backoff_time(curr_node)
                    for i in range(len(curr_node.queue)):
                        frame_time = curr_node.queue[i]
                        # bubble the new receiver_frame_time to all events that have frame_time less than this time
                        if frame_time < new_receiver_frame_time:
                            curr_node.queue[i] = new_receiver_frame_time
                        else:
                            break
                    
                    collision_detected = True
                    max_distance = max(max_distance, distance_to_sender)
            
            curr_index = self.get_next_index(curr_index, direction)

            if direction == "right" and curr_index >= self.N:
                break
            elif direction == "left" and curr_index < 0:
                break


    def get_next_index(curr_index, direction):
        if direction == "right":
            curr_index += 1
        elif direction == "left":
            curr_index -= 1


class NonPersisentCSMACD:
    pass 

if __name__ == "__main__":
    persisentCSMACD = PersisentCSMACD(5, 7, 1000)
    persisentCSMACD.create_nodes()
    
    print(persisentCSMACD.nodes[0].queue)
    
