from typing import List, Tuple, Dict
from collections import deque, defaultdict
from queue import PriorityQueue, Queue
import math
import heapq
class Node:
    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates
        self.g_cost = float('inf')  # Cost from start node to current node
        self.parent = None

    def __lt__(self, other):
        return self.g_cost < other.g_cost

def parse_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        algorithm = lines[0].strip()
        uphill_energy_limit = int(lines[1].strip())
        num_locations = int(lines[2].strip())
        locations = [line.strip().split() for line in lines[3:3+num_locations]]
        num_path_segments = int(lines[3+num_locations].strip())
        segments = [line.strip().split() for line in lines[4+num_locations:4+num_locations+num_path_segments]]

    return algorithm, uphill_energy_limit, locations, segments

def calculate_energy(z1, z2):
    return z2 - z1

def build_graph_with_constraints(locations, path_segments, uphill_energy_limit):
    graph = defaultdict(dict)
    locations_dict = {loc[0]: (int(loc[1]), int(loc[2]), int(loc[3])) for loc in locations}
    
    # Print locations_dict to verify the coordinates
    
    for segment in path_segments:
        start, end = segment
        x1, y1, z1 = locations_dict[start]
        x2, y2, z2 = locations_dict[end]
        
        # Print coordinates to verify correctness
        
        energy_needed = calculate_energy(z1, z2)
        # Calculate 2D Euclidean distance
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        heuristic = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        graph[start][end] = (energy_needed, distance, heuristic)
        graph[end][start] = (-energy_needed, distance, heuristic)  # Assuming undirected graph
    
    return graph

def bfs(graph, start, goal, energy_limit, locations):
    queue = deque([(start, [start], 0)])  # Queue of (node, path, momentum) tuples
    visited = {start: 0}  # Mark the start node as visited with momentum 0

    while queue:
        # Sort the queue based on the number of nodes to the goal

        current_node, path, momentum = queue.popleft()

        if current_node == goal:
            return path
        
        for neighbor, value in graph[current_node].items():
            energy_needed = value[0]
            next_momentum = -energy_needed if energy_needed < 0 else 0
            
            if energy_limit + momentum >= energy_needed:
                if neighbor not in visited or next_momentum > visited.get(neighbor, -float('inf')):
                    visited[neighbor] = next_momentum
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path, next_momentum))

    return None


def calculate_distance(loc1, loc2):
    x1, y1 = loc1
    x2, y2 = loc2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def ucs(graph, start, goal, energy_limit, locations):
    queue = PriorityQueue()
    queue.put((0, start, [start], 0))  # (cost, current node, path, momentum)
    visited = {start: [(0,0)]}  # Mark the start node as visited with cost 0 and momentum 0

    while not queue.empty():
        cost, current_node, path, momentum = queue.get()

        if current_node == goal:
            return path

        for neighbor, value in graph[current_node].items():
            energy_needed = value[0]
            distance = value[1]
            # Calculate next momentum based on the energy of the move
            next_momentum = -energy_needed if energy_needed < 0 else 0

            if energy_limit + momentum >= energy_needed:
                # Calculate the distance between current node and neighbor
                new_cost = cost + distance  # Update cost with Euclidean distance

                # Check if the neighbor has not been visited or if the new cost is lower than the recorded cost
                if neighbor not in visited:
                    visited[neighbor] = [(new_cost, next_momentum)]
                    new_path = path + [neighbor]
                    queue.put((new_cost, neighbor, new_path, next_momentum))
                else: 
                    add_to_queue = True
                    for node_cost, node_momentum in visited[neighbor]:
                        if new_cost >= node_cost and next_momentum <= node_momentum:
                            add_to_queue = False
                            break
                    if add_to_queue:
                        visited[neighbor].append((new_cost, next_momentum))
                        new_path = path + [neighbor]
                        queue.put((new_cost, neighbor, new_path, next_momentum))

    return None
    


# A* algorithm implementation with Euclidean distance as the cost
def a_star(graph, start, goal, energy_limit, locations):
    queue = PriorityQueue()
    queue.put((0, start, [start], 0))  # (cost, current node, path, momentum)
    visited = {start: [(0,0)]}  # Mark the start node as visited with cost 0 and momentum 0

    while not queue.empty():
        cost, current_node, path, momentum = queue.get()

        if current_node == goal:
            return path

        for neighbor, value in graph[current_node].items():
            energy_needed = value[0]
            heuristic = value[2]
            # Calculate next momentum based on the energy of the move
            next_momentum = -energy_needed if energy_needed < 0 else 0

            if energy_limit + momentum >= energy_needed:
                # Calculate the distance between current node and neighbor
                new_cost = cost + heuristic  # Update cost with Euclidean distance

                # Check if the neighbor has not been visited or if the new cost is lower than the recorded cost
                if neighbor not in visited:
                    visited[neighbor] = [(new_cost, next_momentum)]
                    new_path = path + [neighbor]
                    queue.put((new_cost, neighbor, new_path, next_momentum))
                else: 
                    add_to_queue = True
                    for node_cost, node_momentum in visited[neighbor]:
                        if new_cost >= node_cost and next_momentum <= node_momentum:
                            add_to_queue = False
                            break
                    if add_to_queue:
                        visited[neighbor].append((new_cost, next_momentum))
                        new_path = path + [neighbor]
                        queue.put((new_cost, neighbor, new_path, next_momentum))

    return None
  # Return infinity cost if no path found within energy limit

def main():
    input_filename = 'input.txt'
    output_filename = 'output.txt'

    algorithm, energy_limit, locations, segments = parse_input(input_filename)
    
    graph = build_graph_with_constraints(locations, segments, energy_limit)
    
    # Find the start and goal nodes from the locations list
    start = 'start'
    goal = 'goal'
    
    if algorithm == 'BFS':
        path = bfs(graph, start, goal, energy_limit, locations)
    elif algorithm == 'UCS':
        path = ucs(graph, start, goal, energy_limit, locations)
    elif algorithm == 'A*': 
        path = a_star(graph, start, goal, energy_limit, locations)
    with open(output_filename, 'w') as file:
        if path:
            file.write(' '.join(path))
        else:
            file.write('FAIL')


if __name__ == "__main__":
    main()


