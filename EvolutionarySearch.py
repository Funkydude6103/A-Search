import sys
import random

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QHBoxLayout, QLabel
from matplotlib import pyplot as plt, animation


class GridWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.grid_size = 60
        self.grid = [[1 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.start_set = False
        self.goal_set = False

        self.main_layout = QHBoxLayout()

        self.grid_layout = QGridLayout()
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        self.grid_layout.setSpacing(0)

        self.main_layout.addLayout(self.grid_layout)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                button = QPushButton('')
                button.setFixedSize(10, 10)
                if i == 2 and j == 3:
                    button.setStyleSheet("background-color: red")
                elif  i == 38 and j == 39:
                    button.setStyleSheet("background-color: green")
                else:
                    button.setStyleSheet("background-color: black")
                button.clicked.connect(lambda _, i=i, j=j: self.toggle_blockage(i, j))
                self.grid_layout.addWidget(button, i, j)

        self.grid[2][3]=2
        self.grid[38][39]=3


        self.cost_label = QLabel()
        self.main_layout.addWidget(self.cost_label)

        buttons_layout = QGridLayout()

        start_button = QPushButton('Start')
        start_button.clicked.connect(self.set_start)
        buttons_layout.addWidget(start_button, 0, 0)

        goal_button = QPushButton('Goal')
        goal_button.clicked.connect(self.set_goal)
        buttons_layout.addWidget(goal_button, 0, 1)

        solve_button = QPushButton('Solve')
        solve_button.clicked.connect(self.solve_grid)
        buttons_layout.addWidget(solve_button, 1, 0, 1, 2)

        print_button = QPushButton('Print')
        print_button.clicked.connect(self.print_grid)
        buttons_layout.addWidget(print_button, 2, 0, 1, 2)

        self.main_layout.addLayout(buttons_layout)

        self.setLayout(self.main_layout)

        self.setWindowTitle('Evolutionary Search')
        self.setGeometry(100, 100, 900, 800)
        self.show()

    def toggle_blockage(self, i, j):
        if self.start_set == True:
            self.start_set = False
            self.grid[i][j] = 2
            self.sender().setStyleSheet("background-color: red")
            return

        if self.goal_set == True:
            self.goal_set = False
            self.grid[i][j] = 3
            self.sender().setStyleSheet("background-color: green")
            return

        if self.grid[i][j] == 0:
            self.grid[i][j] = 1
            self.sender().setStyleSheet("background-color: black")
            return
        else:
            self.grid[i][j] = 0
            self.sender().setStyleSheet("background-color: white")
            return

    def set_start(self):
        self.start_set = True

    def set_goal(self):
        self.goal_set = True

    def place_blockage(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid[i][j] = 0
                self.grid_layout.itemAtPosition(i, j).widget().setStyleSheet("background-color: white")

    def print_grid(self):
        for row in self.grid:
            print(''.join(map(str, row)))

    def solve_grid(self):
        pass


class EvolutionarySearch(GridWindow):
    def __init__(self):
        super().__init__()

    def solve_grid(self):
        path, steps = self.evolve_path()
        self.color_path(path)
        self.cost_label.setText("Cost: " + str(len(path) - 1) + "\nGenerations: " + str(steps))

    def evolve_path(self):
        # Initialize population
        population = []
        generation = 0
        avg_fitness_values = []
        while len(population) < 30:
            print(len(population))
            new_path = self.random_path()
            if new_path is not None:
                population.append(new_path)

        best_fitness = float('inf')
        stable_count = 0

        for step in range(1000):  # Run evolution for 1000 iterations
            generation += 1
            print("Iteration:", step)
            print("Population:")
            for p in population:
                print(p)
            # Evaluate fitness of each path
            fitness_scores = [self.fitness(path) for path in population]
            best_path = population[fitness_scores.index(min(fitness_scores))]
            current_best_fitness = min(fitness_scores)

            # Check for stability in fitness
            if current_best_fitness >= best_fitness:
                stable_count += 1
            else:
                stable_count = 0
            best_fitness = current_best_fitness

            # If fitness becomes stable, terminate
            if stable_count >= 20:  # Adjust the stability threshold as needed
                break

            # Calculate average fitness
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            avg_fitness_values.append(avg_fitness)

            # Selection
            selected = random.choices(population, weights=fitness_scores, k=10)

            # Crossover
            offspring = [self.crossover(selected[random.randint(0, 9)], selected[random.randint(0, 9)]) for _ in
                         range(6)]

            mutated = [self.mutate_path(selected[random.randint(0, 9)]) for _ in
                       range(6)]

            population = population + offspring + mutated

            # Remove None values from the population
            population = [path for path in population if path is not None]

            # Selecting the Genes to Kill
            sorted_population = sorted(population, key=self.fitness)
            cutoff_index = int(0.8 * len(sorted_population))
            population = sorted_population[:cutoff_index]

        # Plot final graph
        plt.plot(range(len(avg_fitness_values)), avg_fitness_values)
        plt.title('Generation vs Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.show()

        return best_path, step

    def mutate_path(self, chromosome):
        mutated_chromosome = chromosome[:]  # Create a copy of the chromosome to avoid modifying the original
        index_to_mutate = random.randint(0, len(mutated_chromosome) - 1)  # Select a random index to mutate

        current_gene = mutated_chromosome[index_to_mutate]  # Get the current gene at the selected index

        # Find valid adjacent neighbors of the current gene in the path
        valid_neighbors = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (current_gene[0] + dx, current_gene[1] + dy)
            if neighbor in mutated_chromosome and neighbor != current_gene:
                prev_index = (index_to_mutate - 1 + len(mutated_chromosome)) % len(mutated_chromosome)
                next_index = (index_to_mutate + 1) % len(mutated_chromosome)
                if mutated_chromosome[prev_index] == neighbor or mutated_chromosome[next_index] == neighbor:
                    valid_neighbors.append(neighbor)

        if valid_neighbors:
            # Filter valid neighbors to ensure that they form a continuous path
            valid_neighbors = [neighbor for neighbor in valid_neighbors if
                               self.is_continuous_path(mutated_chromosome, index_to_mutate, neighbor)]
            if valid_neighbors:
                new_gene = random.choice(valid_neighbors)
                mutated_chromosome[index_to_mutate] = new_gene

        return mutated_chromosome

    def is_continuous_path(self, chromosome, index, candidate_gene):
        # Check if inserting the candidate gene at the given index forms a continuous path
        # We'll check if the chromosome is continuous after inserting the candidate gene
        temp_chromosome = chromosome[:index] + [candidate_gene] + chromosome[index + 1:]
        for i in range(len(temp_chromosome) - 1):
            if abs(temp_chromosome[i][0] - temp_chromosome[i + 1][0]) + abs(
                    temp_chromosome[i][1] - temp_chromosome[i + 1][1]) != 1:
                return False
        return True

    def random_path(self):
        start = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 2][0]
        goal = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 3][0]

        def heuristic(point):
            # Manhatten distance as heuristic (h)
            return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

        path = [start]
        current = start
        max_iterations = self.grid_size * self.grid_size * self.grid_size  # Maximum number of iterations allowed

        while current != goal and max_iterations > 0:
            possible_moves = [(current[0] + 1, current[1]), (current[0] - 1, current[1]),
                              (current[0], current[1] + 1), (current[0], current[1] - 1)]
            possible_moves.sort(key=lambda x: heuristic(x) + len(path))  # Sort moves based on g + h

            # Filter out moves that are out of bounds or on obstacles
            possible_moves = [move for move in possible_moves if
                              0 <= move[0] < self.grid_size and 0 <= move[1] < self.grid_size and
                              self.grid[move[0]][move[1]] != 0]

            next_move = random.choice(possible_moves[:2]) if possible_moves else None
            if next_move and next_move not in path:
                path.append(next_move)
                current = next_move
            max_iterations -= 1

        if current != goal:  # If goal is not reached, return None
            print("none")
            return None

        return path

    def fitness(self, path):
        goal = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 3][0]
        return len(path) - abs(path[-1][0] - goal[0]) - abs(path[-1][1] - goal[1])

    def crossover(self, path1, path2):
        # Ensure that crossover point does not break the integrity of paths
        common_points = list(set(path1) & set(path2))
        if not common_points:
            return path1

        common_point = random.choice(common_points)
        index1 = path1.index(common_point)
        index2 = path2.index(common_point)

        return path1[:index1] + path2[index2:]


    def color_path(self, path):
        start = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 2][0]
        goal = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 3][0]

        for position in path[1:-1]:  # Exclude start and end points
            if position != start and position != goal:
                self.grid_layout.itemAtPosition(position[0], position[1]).widget().setStyleSheet(
                    "background-color: blue")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EvolutionarySearch()
    sys.exit(app.exec_())
