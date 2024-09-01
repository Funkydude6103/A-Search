import sys
import random
import heapq

from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QHBoxLayout, QLabel


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
                elif i == 38 and j == 39:
                    button.setStyleSheet("background-color: green")
                else:
                    button.setStyleSheet("background-color: black")
                button.clicked.connect(lambda _, i=i, j=j: self.toggle_blockage(i, j))
                self.grid_layout.addWidget(button, i, j)

        self.grid[2][3] = 2
        self.grid[38][39] = 3

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

        self.setWindowTitle('A* Search')
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


class AStar(GridWindow):
    def __init__(self):
        super().__init__()

    def solve_grid(self):
        start, goal = self.find_start_and_goal()
        path, cost = self.a_star(start, goal)
        self.color_path(path)
        self.cost_label.setText("Cost: " + str(cost))

    def find_start_and_goal(self):
        start = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 2][0]
        goal = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 3][0]
        return start, goal

    def a_star(self, start, goal):
        open_set = []
        closed_set = set()
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            current_cost, current_node = heapq.heappop(open_set)

            if current_node == goal:
                path = self.reconstruct_path(came_from, current_node)
                return path, g_score[current_node]

            closed_set.add(current_node)

            for neighbor in self.get_neighbors(current_node):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current_node] + 1

                if neighbor not in [node[1] for node in open_set] or tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return [], float('inf')

    def get_neighbors(self, node):
        i, j = node
        neighbors = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
        valid_neighbors = [(x, y) for x, y in neighbors if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x][y] != 0]
        return valid_neighbors

    def heuristic(self, node, goal):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def color_path(self, path):
        start = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 2][0]
        goal = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if self.grid[i][j] == 3][0]

        for position in path[1:-1]:
            if position != start and position != goal:
                self.grid_layout.itemAtPosition(position[0], position[1]).widget().setStyleSheet(
                    "background-color: blue")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AStar()
    sys.exit(app.exec_())
