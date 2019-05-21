import csv
import heapq
import collections

# Defining the maze characters
WALL = '#'
START = 'S'
END = 'E'
PATH = ' '
SOLUTION = '.'

# Using the Manhattan distance (sum of the absolute differences of the coordinates) as heuristic.
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* Search Algorithm ---
def a_star_search(maze):
    """
    Finds the shortest path from 'S' to 'E' using the A* algorithm.
    A* is a best-first search that uses a heuristic function to guide its search.
    It is more efficient than BFS because it prioritizes exploring nodes that
    appear to be closer to the goal. It guarantees the shortest path.
    """
    start = None
    end = None
    rows, cols = len(maze), len(maze[0])
    
    # Find start and end points
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == START:
                start = (r, c)
            elif maze[r][c] == END:
                end = (r, c)

    if not start or not end:
        return None, None

    # Priority queue stores tuples of (f_score, g_score, position, path)
    # f_score = g_score + h_score
    # g_score = cost from start to current position
    # h_score = heuristic estimate of cost from current to goal
    open_set = [(heuristic(start, end), 0, start, [start])]
    
    # g_score dictionary to keep track of the cost from start to each node
    g_scores = {start: 0}
    
    while open_set:
        # Get the node with the lowest f_score
        f_score, g_score, current, path = heapq.heappop(open_set)

        if current == end:
            return path, len(path) - 1

        # Explore neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)

            # Check if the neighbor is valid
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Check for walls
                if maze[neighbor[0]][neighbor[1]] != WALL:
                    # Calculate new g_score for the neighbor
                    new_g_score = g_score + 1

                    # If this is a better path to the neighbor
                    if new_g_score < g_scores.get(neighbor, float('inf')):
                        g_scores[neighbor] = new_g_score
                        h_score = heuristic(neighbor, end)
                        f_score = new_g_score + h_score
                        
                        new_path = path + [neighbor]
                        heapq.heappush(open_set, (f_score, new_g_score, neighbor, new_path))
    
    return None, None

# Breadth-First Search (BFS) Test
def bfs_search(maze):
    """
    Finds the shortest path from 'S' to 'E' using BFS.
    BFS explores the maze layer by layer, guaranteeing the shortest path
    in terms of the number of steps. It is an uninformed search,
    meaning it does not use a heuristic to guide its search.
    """
    start = None
    end = None
    rows, cols = len(maze), len(maze[0])
    
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == START:
                start = (r, c)
            elif maze[r][c] == END:
                end = (r, c)

    if not start or not end:
        return None, None

    queue = collections.deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if current == end:
            return path, len(path) - 1
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                maze[neighbor[0]][neighbor[1]] != WALL and neighbor not in visited):
                
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
    
    return None, None

# Depth-First Search (DFS) Test
def dfs_search(maze):
    """
    Finds a path from 'S' to 'E' using DFS.
    DFS explores as deeply as possible along each branch before backtracking.
    It uses a stack (or recursion). It is an uninformed search and does NOT
    guarantee the shortest path.
    """
    start = None
    end = None
    rows, cols = len(maze), len(maze[0])
    
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == START:
                start = (r, c)
            elif maze[r][c] == END:
                end = (r, c)

    if not start or not end:
        return None, None

    stack = [(start, [start])]
    visited = {start}
    
    while stack:
        current, path = stack.pop()
        
        if current == end:
            return path, len(path) - 1
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                maze[neighbor[0]][neighbor[1]] != WALL and neighbor not in visited):
                
                visited.add(neighbor)
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))
    
    return None, None

def load_maze(filename):
    """Loads a maze from a CSV file."""
    maze = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # Remove any whitespace from cells
                maze.append([cell.strip() for cell in row])
        return maze
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None

def print_solution(maze, path, algorithm_name):
    """Prints the maze with the solution path highlighted."""
    if not path:
        print(f"No solution found for '{algorithm_name}'.")
        return
    
    solution_maze = [list(row) for row in maze]
    for r, c in path:
        if solution_maze[r][c] not in [START, END]:
            solution_maze[r][c] = SOLUTION
    
    print(f"\n--- Solved Maze using {algorithm_name} ---")
    for row in solution_maze:
        print(" ".join(row))

def main():
    """Main function to run the maze solver."""
    maze_files = ["maze1.csv", "maze2.csv", "maze3.csv"]
    
    for filename in maze_files:
        maze = load_maze(filename)
        if not maze:
            continue
        
        # Original maze print
        print("\n--- Original Maze ---")
        for row in maze:
            print(" ".join(row))
        
        # --- Run A* Search ---
        path_a_star, length_a_star = a_star_search(maze)
        if path_a_star:
            print_solution(maze, path_a_star, "A* Search")
            print(f"Path length: {length_a_star} steps")
            print("A* guarantees the shortest path by using a heuristic to guide the search, making it very efficient.")
        else:
            print("\n--- No solution found by A* Search ---")
            
        # --- Run BFS ---
        path_bfs, length_bfs = bfs_search(maze)
        if path_bfs:
            print_solution(maze, path_bfs, "Breadth-First Search (BFS)")
            print(f"Path length: {length_bfs} steps")
            print("BFS guarantees the shortest path but explores the maze layer by layer, which can be less efficient than A*.")
        else:
            print("\n--- No solution found by BFS ---")

        # --- Run DFS ---
        path_dfs, length_dfs = dfs_search(maze)
        if path_dfs:
            print_solution(maze, path_dfs, "Depth-First Search (DFS)")
            print(f"Path length: {length_dfs} steps")
            print("DFS finds a path quickly but does NOT guarantee the shortest path. It explores deeply before backtracking.")
        else:
            print("\n--- No solution found by DFS ---")

if __name__ == "__main__":
    main()
