from typing import List, Union
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import deque

print(os.listdir())
os.chdir(os.path.dirname(__file__))
def extract_data(TSP_file_path: str) -> np.ndarray:
    x_data: List = []
    y_data: List = []
    reading_coords = False
    with open(TSP_file_path, 'r') as file:
        for line in file.readlines():
            if line == 'NODE_COORD_SECTION\n':
                reading_coords = True
                continue
            if line == 'EOF\n':
                break
            if reading_coords:
                #print(line)
                nums = line.strip().split(' ')
                x_data.append(int(nums[1]))
                y_data.append(int(nums[2]))
    x_coords = np.array(x_data)
    y_coords = np.array(y_data)
    coord_matrix = np.array((x_coords, y_coords))
    distance_matrix = np.rint(np.sqrt((x_coords - x_coords[:, np.newaxis])**2 + (y_coords - y_coords[:, np.newaxis])**2))
    return distance_matrix, coord_matrix

def greedy(distance_mx: np.ndarray, first_element: int = None):
    indices_mx = np.arange(100)
    if first_element is None:
        first_element = np.random.randint(0, 100)
    route = [first_element]
    element_is_available = np.full(100, True)
    element_is_available[first_element] = False
    front_edge = first_element
    back_edge = first_element
    total_dist = 0
    for i in range(48):
        a = distance_mx[:, front_edge]
        shortest_dist = np.min(a[element_is_available])
        new_element = indices_mx[element_is_available][np.argmin(a[element_is_available])]
        route.append(new_element)
        element_is_available[new_element] = False
        front_edge = new_element
        total_dist += shortest_dist
    b = distance_mx[:, front_edge] + distance_mx[:, back_edge]
    candidates = b[element_is_available]
    shortest_dist = np.min(candidates)
    new_element = indices_mx[element_is_available][np.argmin(b[element_is_available])]
    route.append(new_element)
    route.append(first_element)
    total_dist += shortest_dist
    return total_dist, route


#def include_in_cycle(index, new_element):


def greedy_cycle(distance_mx: np.ndarray, first_element: int = None):
    indices_mx = np.arange(100)
    if first_element is None:
        first_element = np.random.randint(0, 100)
    route = [first_element, first_element]
    neighbor_distances = [0]
    element_available = np.full(100, True)
    element_available[first_element] = False
    #total_dist = 0
    for i in range(49):
        # otrzymanie 2-wymiarowej macierzy NxK kosztów włączenia punktu k w miejsce n cyklu
        insertion_cost = distance_mx[route[:-1]][:, element_available] + distance_mx[route[1:]][:, element_available] - np.array(neighbor_distances)[:, np.newaxis]
        if insertion_cost.ndim == 1:
            insertion_cost = insertion_cost[:, np.newaxis]
        # otrzymanie indeksów najlepszego przyłączenia: odpowiednio miejsca przyłączenia i przyłączanego elementu
        #total_dist += np.min(insertion_cost)
        cheapest_insertion = np.unravel_index(np.argmin(insertion_cost), insertion_cost.shape)
        insertion_position = cheapest_insertion[0]+1
        inserted_element = indices_mx[element_available][cheapest_insertion[1]]
        # wstawienie elementu do cyklu Hamiltona
        route.insert(insertion_position, inserted_element)
        # zastąpienie długości starej krawędzi dwiema długosciami nowych krawędzi
        neighbor_distances[insertion_position-1] = distance_mx[route[insertion_position+1], inserted_element]
        neighbor_distances.insert(insertion_position-1, distance_mx[route[insertion_position-1], inserted_element])
        # włączony element nie jest rozważany w kolejnych cyklach
        element_available[inserted_element] = False
    total_dist = sum(neighbor_distances)
    return total_dist, route


def two_regret(distance_mx: np.ndarray, first_element: int = None):
    indices_mx = np.arange(100)
    if first_element is None:
        first_element = np.random.randint(0, 100)
    route = [first_element, first_element]
    neighbor_distances = [0]
    element_available = np.full(100, True)
    element_available[first_element] = False
    # w 1. i 2. iteracji, gdy jest tylko 1 sposób przyłączenia, algorytm działa jak greedy cycle, następnie działa na zasadzie 2-żalu
    for i in range(49):
        insertion_cost = distance_mx[route[:-1]][:, element_available] + distance_mx[route[1:]][:, element_available] - np.array(neighbor_distances)[:, np.newaxis]
        if i < 2:
            insertion_cost = insertion_cost[:, np.newaxis]
            chosen_insertion = np.unravel_index(np.argmin(insertion_cost), insertion_cost.shape)
        if i >= 2:
            regret_submatrix = np.partition(insertion_cost, 1, axis=0)[:2, :]
            regret_vector = regret_submatrix[1] - regret_submatrix[0]
            highest_regret_index = np.argmax(regret_vector)
            chosen_insertion = (np.argmin(insertion_cost[:, highest_regret_index]), highest_regret_index)
        insertion_position = chosen_insertion[0]+1
        inserted_element = indices_mx[element_available][chosen_insertion[1]]
        route.insert(insertion_position, inserted_element)
        neighbor_distances[insertion_position-1] = distance_mx[route[insertion_position+1], inserted_element]
        neighbor_distances.insert(insertion_position-1, distance_mx[route[insertion_position-1], inserted_element])
        element_available[inserted_element] = False
    total_dist = sum(neighbor_distances)
    return total_dist, route

def weighted_regret(distance_mx: np.ndarray, first_element: int = None, k:int =0.5):
    indices_mx = np.arange(100)
    if first_element is None:
        first_element = np.random.randint(0, 100)
    route = [first_element, first_element]
    neighbor_distances = [0]
    element_available = np.full(100, True)
    element_available[first_element] = False
    for i in range(49):
        insertion_cost = distance_mx[route[:-1]][:, element_available] + distance_mx[route[1:]][:, element_available] - np.array(neighbor_distances)[:, np.newaxis]
        if i < 2:
            insertion_cost = insertion_cost[:, np.newaxis]
            chosen_insertion = np.unravel_index(np.argmin(insertion_cost), insertion_cost.shape)
        if i >= 2:
            regret_submatrix = np.partition(insertion_cost, 1, axis=0)[:2, :]
            regret_vector = regret_submatrix[1] - regret_submatrix[0]
            weighted_insertion_cost = (1-k)*insertion_cost - k*regret_vector
            chosen_insertion = np.unravel_index(np.argmin(weighted_insertion_cost), weighted_insertion_cost.shape)
        insertion_position = chosen_insertion[0]+1
        inserted_element = indices_mx[element_available][chosen_insertion[1]]
        route.insert(insertion_position, inserted_element)
        neighbor_distances[insertion_position-1] = distance_mx[route[insertion_position+1], inserted_element]
        neighbor_distances.insert(insertion_position-1, distance_mx[route[insertion_position-1], inserted_element])
        element_available[inserted_element] = False
    total_dist = sum(neighbor_distances)
    return total_dist, route



tsp1 = 'kroA100.tsp'
tsp2 = 'kroB100.tsp'

dist1, coord1 = extract_data(tsp1)
dist2, coord2 = extract_data(tsp2)

two_regret(dist1)

def point_distance(x1, x2, y1, y2):
    return int((((x1-x2)**2)+((y1-y2)**2))**(1/2))

starting_points = np.arange(100)
np.random.shuffle(starting_points)

np.random.seed(6543)

for dataset in ((dist1, coord1), (dist2, coord2)):
    for method, name in ((greedy, "greedy"),(greedy_cycle, "greedy_cycle"), (two_regret, "2-regret"), (weighted_regret, "weighted regret")):
        minn = 10**6
        maxx = 0
        total = 0
        best_route = []
        for point in starting_points[:50]:
            dist, route = method(dataset[0], point)
            if dist < minn:
                minn = dist
                best_route = route
            total += dist
            if dist > maxx:
                maxx = dist
        avg = total/50
        plt.plot(dataset[1][0], dataset[1][1], "o")
        plt.plot(dataset[1][0][best_route], dataset[1][1][best_route])
        plt.show()
        print(name, minn, avg, maxx)