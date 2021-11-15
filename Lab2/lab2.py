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

def plot_graphs(Ns, Es):
    plt.figure(1)
    plt.plot(Ns, Es)
    plt.title("Efficiency vs. Nodes")
    plt.ylabel('Efficiency of system (successful packets/transmitted packets)')
    plt.xlabel('Number of nodes')
    plt.show()

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
        self.total_transmissions = 0
        self.successful_packet_transmissions = 0
        self.nodes = []

    def create_nodes(self):
        for _ in range(self.N):
            node = Node(self.A, self.T)
            node.create_queue()
            self.nodes.append(node)

    def run_simulation(self):
        # check which arrival time is the smallest among the nodes and get node index
        while self.completed_nodes < self.N:
            min_node_index = 0
            min_timestamp = float('inf')
            for i, node in enumerate(self.nodes):
                if node.queue and node.queue[0] < min_timestamp:
                    min_timestamp = node.queue[0]
                    min_node_index = i

            # process event at node index
            self.process_sender_node(min_node_index, min_timestamp)
    
    def calculate_exp_backoff_time(self, node):
        Tp = 512 / self.R
        return random.randint(0, (2**node.backoff_counter) - 1) * Tp

    def bubble_new_frame_time(self, node, new_frame_time):
        for i in range(len(node.queue)):
            current_frame_time = node.queue[i]
            if current_frame_time < new_frame_time:
                node.queue[i] = new_frame_time
            else:
                break

    def process_sender_node(self, sender_index, sender_frame_time):
        collision_status = { "collision_detected": False, "max_distance": float('-inf') }
        self.total_transmissions += 1

        # send packet to nodes right of the sender
        self.process_neighbour_node(sender_index+1, sender_frame_time, collision_status, direction="right")

        # send packet to nodes left of the sender
        self.process_neighbour_node(sender_index-1, sender_frame_time, collision_status, direction="left")

        sender_node = self.nodes[sender_index]

        if collision_status['collision_detected']:
            sender_node.backoff_counter += 1
            if sender_node.should_drop_packet():
                sender_node.drop_packet()
                self.dropped_packets += 1
            else:
                wait_time = sender_frame_time + compute_propagation_delay(self.D, self.S)*collision_status['max_distance'] + self.calculate_exp_backoff_time(sender_node)
                self.bubble_new_frame_time(sender_node, wait_time)
        else:
            sender_node.queue.popleft()
            self.successful_packet_transmissions += 1
            transmission_time = sender_frame_time + compute_transmission_delay(self.L, self.R)
            self.bubble_new_frame_time(sender_node, transmission_time)
            
        if not sender_node.queue:
            self.completed_nodes += 1

    def process_neighbour_node(self, sender_index, sender_frame_time, collision_status, direction):
        curr_index = sender_index
        max_distance = collision_status["max_distance"]

        while curr_index >= 0 and curr_index < self.N:
            curr_node = self.nodes[curr_index]
            # if the current node has no more events to offer in its queue, go next
            if not curr_node.queue:
                curr_index = self.get_next_index(curr_index, direction)
                continue
            
            head_frame_time = curr_node.queue[0]
            distance_to_sender = abs(curr_index - sender_index)
            first_bit_received_time = sender_frame_time + (compute_propagation_delay(self.D, self.S) * (distance_to_sender))

            # collision occurs
            if head_frame_time <= first_bit_received_time:
                self.total_transmissions += 1
                curr_node.backoff_counter += 1

                if curr_node.should_drop_packet():
                    curr_node.drop_packet()
                    self.dropped_packets += 1
                    if not curr_node.queue:
                        self.completed_nodes += 1
                else:
                    wait_time = first_bit_received_time + compute_transmission_delay(self.L, self.R) + self.calculate_exp_backoff_time(curr_node)
                    self.bubble_new_frame_time(curr_node, wait_time)
                    
                collision_status["collision_detected"] = True
                collision_status["max_distance"] = max(max_distance, distance_to_sender)

            # no collision but node senses medium as busy
            if head_frame_time > first_bit_received_time and head_frame_time < first_bit_received_time + compute_transmission_delay(self.L, self.R):
                transmission_time = first_bit_received_time + compute_transmission_delay(self.L, self.R)
                self.bubble_new_frame_time(curr_node, transmission_time)
            
            curr_index = self.get_next_index(curr_index, direction)

    def get_next_index(self, curr_index, direction):
        if direction == "right":
            return curr_index + 1
        elif direction == "left":
            return curr_index - 1


class NonPersisentCSMACD:
    pass 

if __name__ == "__main__":
    Ns = [20, 40, 60, 80, 100]
    As = [7, 10, 20]
    efficiency = []
    for N in Ns:
        persisentCSMACD = PersisentCSMACD(N, 12, 1000)
        persisentCSMACD.create_nodes()
        persisentCSMACD.run_simulation()
        efficiency.append(persisentCSMACD.successful_packet_transmissions/persisentCSMACD.total_transmissions)
    plot_graphs(Ns, efficiency)
        # print(persisentCSMACD.successful_packet_transmissions)
        # print(persisentCSMACD.total_transmissions)
        # print(persisentCSMACD.completed_nodes)
        # print(persisentCSMACD.dropped_packets)
        # print(">>>>>>")
        # print(persisentCSMACD.successful_packet_transmissions/persisentCSMACD.total_transmissions)
