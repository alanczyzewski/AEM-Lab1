import os
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools


def extract_data(TSP_file_path: str) -> np.ndarray:
    x_data = []
    y_data = []
    reading_coords = False
    with open(TSP_file_path, 'r') as file:
        for line in file.readlines():
            if line == 'NODE_COORD_SECTION\n':
                reading_coords = True
                continue
            if line == 'EOF\n':
                break
            if reading_coords:
                nums = line.strip().split(' ')
                x_data.append(int(nums[1]))
                y_data.append(int(nums[2]))
    x_coords = np.array(x_data)
    y_coords = np.array(y_data)
    coord_matrix = np.array((x_coords, y_coords))
    distance_matrix = np.rint(np.sqrt((x_coords - x_coords[:, np.newaxis])**2 + (y_coords - y_coords[:, np.newaxis])**2))
    return distance_matrix, coord_matrix


# ruchy zmieniajace zbior wierzcholkow - greedy

def token_generator_edges():
    swap_tokens = tuple([(0, i, j) for i in range(50) for j in range(50)])
    inverse_tokens = tuple([(1, i, j) for i in range(50) for j in range(2,25)])
    symmetrical_inverse_tokens = tuple([[1, i, 25] for i in range(25)])
    tokens = swap_tokens + inverse_tokens + symmetrical_inverse_tokens
    token_numbers = np.arange(len(tokens))
    np.random.shuffle(token_numbers)
    for n in token_numbers:
        yield tokens[n]


def greedy_edges(distance_mx: np.ndarray, route):
    distance_is_improving = True
    not_in_route = np.array([i for i in range(100) if i not in route])
    while distance_is_improving:
        distance_is_improving = False
        for token in token_generator_edges():
            if token[0] == 0:
                j, k = token[1], token[2]
                old_dist = distance_mx[route[(j - 1) % 50], route[j]] + distance_mx[route[j], route[(j + 1) % 50]]
                new_dist = distance_mx[route[(j - 1) % 50], not_in_route[k]] + distance_mx[not_in_route[k], route[(j + 1) % 50]]
                delta = new_dist - old_dist
                if delta < 0:
                    new_vertex = not_in_route[k]
                    not_in_route[k] = route[j]
                    route[j] = new_vertex
                    distance_is_improving = True
            if token[0] == 1:
                j, k = token[1], token[2]
                old_dist = distance_mx[route[(j - 1) % 50], route[j]] + distance_mx[
                    route[(j + k - 1) % 50], route[(j + k) % 50]]
                new_dist = distance_mx[route[(j - 1) % 50], route[(j + k - 1) % 50]] + distance_mx[
                    route[j], route[(j + k) % 50]]
                delta = new_dist - old_dist
                if delta < 0:
                    route = np.roll(route, -j)
                    route[:k] = route[k - 1::-1]
                    route = np.roll(route, j)
                    distance_is_improving = True
    return route

# ruchy zmieniajace zbior wierzcholkow - steepest
def steepest_edges(distance_mx: np.ndarray, route):
    while True:
        not_in_route = np.array([i for i in range(100) if i not in route])
        best_swap_delta = np.inf
        for j in range(50):
            for k in range(50):
                old_dist = distance_mx[route[(j-1)%50], route[j]] + distance_mx[route[j], route[(j+1)%50]]
                new_dist = distance_mx[route[(j-1)%50], not_in_route[k]] + distance_mx[not_in_route[k], route[(j+1)%50]]
                new_delta = new_dist - old_dist
                if new_delta < best_swap_delta:
                    best_swap_delta = new_delta
                    best_swap = (j, k)
        best_inversion_delta = np.inf
        for j in range(50):
            for k in range(2, 26):   # for k == 25, elements are iterated twice. That was not addressed to keep the code consise
                old_dist = distance_mx[route[(j-1)%50], route[j]] + distance_mx[route[(j+k-1)%50], route[(j+k)%50]]
                new_dist = distance_mx[route[(j-1)%50], route[(j+k-1)%50]] + distance_mx[route[j], route[(j+k)%50]]
                new_delta = new_dist - old_dist
                if new_delta < best_inversion_delta:
                    best_inversion_delta = new_delta
                    best_inversion = (j, k)
        if best_swap_delta <= best_inversion_delta and best_swap_delta < 0:
            new_vertex = not_in_route[best_swap[1]]
            not_in_route[best_swap[1]] = route[best_swap[0]]
            route[best_swap[0]] = new_vertex
        elif best_inversion_delta < best_swap_delta and best_inversion_delta < 0:
            route = np.roll(route, -best_inversion[0])
            route[:best_inversion[1]] = route[best_inversion[1]-1::-1]
            route = np.roll(route, best_inversion[0])
        else:
            break
    return route

def token_generator_vertices():
    swap_tokens = tuple([(0, i, j) for i in range(50) for j in range(50)])
    inverse_tokens = tuple([(1, i, j) for i, j in itertools.combinations(range(50), 2) if abs(i-j)!=1 and abs(i-j)!=49])
    tokens = swap_tokens + inverse_tokens
    token_numbers = np.arange(len(tokens))
    np.random.shuffle(token_numbers)
    for n in token_numbers:
        yield tokens[n]


def greedy_vertices(distance_mx: np.ndarray, route):
    distance_is_improving = True
    not_in_route = np.array([i for i in range(100) if i not in route])
    while distance_is_improving:
        distance_is_improving = False
        for token in token_generator_vertices():
            if token[0] == 0:
                j, k = token[1], token[2]
                old_dist = distance_mx[route[(j - 1) % 50], route[j]] + distance_mx[route[j], route[(j + 1) % 50]]
                new_dist = distance_mx[route[(j - 1) % 50], not_in_route[k]] + distance_mx[not_in_route[k], route[(j + 1) % 50]]
                delta = new_dist - old_dist
                if delta < 0:
                    new_vertex = not_in_route[k]
                    not_in_route[k] = route[j]
                    route[j] = new_vertex
                    distance_is_improving = True
            if token[0] == 1:
                j, k = token[1], token[2]
                old_dist1 = distance_mx[route[(j - 1) % 50], route[j]] + distance_mx[route[j], route[(j + 1) % 50]]
                old_dist2 = distance_mx[route[(k - 1) % 50], route[k]] + distance_mx[route[k], route[(k + 1) % 50]]
                new_dist1 = distance_mx[route[(j - 1) % 50], route[k]] + distance_mx[
                    route[k], route[(j + 1) % 50]]
                new_dist2 = distance_mx[route[(k - 1) % 50], route[j]] + distance_mx[
                    route[j], route[(k + 1) % 50]]
                delta = new_dist1 + new_dist2 - old_dist1 - old_dist2
                if delta < 0:
                    distold = distance(distance_mx, route)
                    route[[j, k]] = route[[k, j]]
                    distnew = distance(distance_mx, route)
                    truedelta = distnew-distold
                    if truedelta > 1:
                        print(token)
                    distance_is_improving = True
    return route

# ruchy zmieniajace zbior wierzcholkow - steepest
def steepest_vertices(distance_mx: np.ndarray, route):
    while True:
        not_in_route = np.array([i for i in range(100) if i not in route])
        best_swap_delta = np.inf
        for j in range(50):
            for k in range(50):
                old_dist = distance_mx[route[(j-1)%50], route[j]] + distance_mx[route[j], route[(j+1)%50]]
                new_dist = distance_mx[route[(j-1)%50], not_in_route[k]] + distance_mx[not_in_route[k], route[(j+1)%50]]
                new_delta = new_dist - old_dist
                if new_delta < best_swap_delta:
                    best_swap_delta = new_delta
                    best_swap = (j, k)
        best_inner_delta = np.inf
        for j, k in itertools.combinations(range(50), 2):
            if abs(j-k)!=1 and abs(j-k)!=49:
                old_dist1 = distance_mx[route[(j - 1) % 50], route[j]] + distance_mx[route[j], route[(j + 1) % 50]]
                old_dist2 = distance_mx[route[(k - 1) % 50], route[k]] + distance_mx[route[k], route[(k + 1) % 50]]
                new_dist1 = distance_mx[route[(j - 1) % 50], route[k]] + distance_mx[route[k], route[(j + 1) % 50]]
                new_dist2 = distance_mx[route[(k - 1) % 50], route[j]] + distance_mx[
                    route[j], route[(k + 1) % 50]]
                new_delta = new_dist1 + new_dist2 - old_dist1 - old_dist2
        if best_swap_delta <= best_inner_delta and best_swap_delta < 0:
            new_vertex = not_in_route[best_swap[1]]
            not_in_route[best_swap[1]] = route[best_swap[0]]
            route[best_swap[0]] = new_vertex
        elif best_inner_delta < best_swap_delta and best_inner_delta < 0:
            route[[best_inner[0], best_inner[1]]] = route[[best_inner[1], best_inner[0]]]
        else:
            break
    return route

def distance(distance_mx, route):
    dist = 0
    for i in range(route.shape[0]):
        dist += distance_mx[route[i], route[(i+1)%50]]
    return dist

# ruchy zmieniajace ruchy wewnatrztrasowe - greedy
    
    # TODO losowac kolejnosc przegladania ruchow wewnatrztrasowych i
    #      zmieniajacych zbior wierzcholkow



os.chdir(os.path.dirname(__file__))

tsp1 = 'kroA100.tsp'
tsp2 = 'kroB100.tsp'

dist1, coord1 = extract_data(tsp1)
dist2, coord2 = extract_data(tsp2)

# TODO start point - random search - generujemy losowe rozwiazania w petli i zwracamy 
#      najlepsze z nich.  Algorytm uruchamiamy na czas taki jak sredni czas 
#      najwolniejszej wersji lokalnego przeszukiwania

# 100 best random solutions
random_routes1 = []
random_routes2 = []
random_route_distances1 = np.empty(100, dtype="int")
random_route_distances2 = np.empty(100, dtype="int")
indices = np.arange(100)
for i in range(10000):
    np.random.shuffle(indices)
    route = np.array(indices[:50])
    random_dist1 = distance(dist1, route)
    random_dist2 = distance(dist2, route)
    if len(random_routes1) < 100:
        random_routes1.append(route)
        random_routes2.append(route)
        random_route_distances1[i] = random_dist1
    else:
        if random_dist1 < np.max(random_route_distances1):
            index = np.argmax(random_route_distances1)
            random_route_distances1[index] = random_dist1
            random_routes1[index] = route
        if random_dist2 < np.max(random_route_distances2):
            index = np.argmax(random_route_distances2)
            random_route_distances2[index] = random_dist2
            random_routes2[index] = route




times = dict()
for distan, coord, random_routes in ((dist1, coord1, random_routes1), (dist2, coord2, random_routes2)):
    for method, name in ((greedy_vertices, "greedy vertices"), (steepest_vertices, "steepest vertices"), (greedy_edges, "greedy edges"), (steepest_edges, "steepest edges")):
        min_dist = np.inf
        max_dist = 0
        total_dist = 0
        min_time = np.inf
        max_time = 0
        total_time = 0
        best_route = []
        for r in random_routes:
            starting_time = time.time()
            route = method(distan, r)
            running_time = time.time() - starting_time
            route_distance = distance(distan, route)

            total_dist += route_distance
            if route_distance < min_dist:
                min_dist = route_distance
                best_route = route
            if route_distance > max_dist:
                max_dist = route_distance

            total_time += running_time
            if running_time < min_time:
                min_time = running_time
            if running_time > max_time:
                max_time = running_time
        avg_dist = total_dist/100
        avg_time = total_time/100
        plt.plot(coord[0], coord[1], "o")
        plt.plot(coord[0][list(best_route)+[best_route[0]]], coord[1][list(best_route)+[best_route[0]]])
        plt.show()
        print(name, min_dist, avg_dist, max_dist)
        print(name + ' times:', min_time, avg_time, max_time)
        # TODO obliczyc min, avg, max dla czasu


# TODO implementacja musi wykorzystywac obliczanie delty funkcji celu

# TODO kazdy algorytm uruchomic 100 razy startujac z rozwiazan losowych
# TODO wartosci min, max, srednia dla funkcji celu i czasu
# TODO wizualizacje
# TODO w sprawozdaniu opisac sposob randomizacji

