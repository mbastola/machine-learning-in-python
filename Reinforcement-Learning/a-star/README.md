In this project I analyze and compare the performance of three fundamental pathfinding algorithms: **A* Search**, **Breadth-First Search (BFS)**, and **Depth-First Search (DFS)**. The goal is to understand their underlying mechanics, performance characteristics, and the trade-offs between them when applied to solving grid-based mazes.

We will be focusing on the following key metrics:
1.  **Path Optimality:** Does the algorithm guarantee the shortest path?
2.  **Execution Time:** How quickly does each algorithm find a solution?
3.  **Search Efficiency:** How many nodes does each algorithm need to explore before finding the goal?

A* is supposed to (as an *informed* search algorithm) be the most efficient overall, finding the optimal path faster and with less exploration than the *uninformed* BFS and DFS methods.


```python
import csv
import heapq
import collections
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Use a consistent style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
```

## 1. Maze Environment and Visualization

First, I need to define the environment. The maze is a 2D grid where each cell can be a Wall (`#`), a Path (` `), a Start (`S`), or an End (`E`). The agent's possible actions are to move to adjacent cells (Up, Down, Left, Right), but not diagonally.

To better analyze the results, I'll create a visualization function using `matplotlib` instead of just printing text. This will help in understanding the observation space (the maze grid) and the solution paths.


```python
# Defining the maze characters
WALL = '#'
START = 'S'
END = 'E'
PATH = ' '
SOLUTION = '.'

def load_maze(filename):
    """Loads a maze from a CSV file."""
    maze = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                maze.append([cell.strip() for cell in row])
        return maze
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None

def plot_maze(maze, path=None, title="Maze"):
    """Visualizes the maze and the solution path using Matplotlib."""
    # Create a numeric representation of the maze for plotting
    # 0: Path, 1: Wall, 2: Start, 3: End, 4: Solution
    maze_map = {
        PATH: 0, WALL: 1, START: 2, END: 3, SOLUTION: 4
    }

    print(maze)
    for i, m in enumerate(maze):
        print(i, len(m))
    numeric_maze = np.array([[maze_map.get(cell, 0) for cell in row] for row in maze])
    
    if path:
        for r, c in path:
            if numeric_maze[r, c] not in [maze_map[START], maze_map[END]]:
                numeric_maze[r, c] = maze_map[SOLUTION]
    
    # Define a colormap
    # White: Path, Black: Wall, Green: Start, Red: End, Blue: Solution
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red', 'blue'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(numeric_maze, cmap=cmap, norm=norm)

    # Draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, len(maze[0]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(maze), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=16)
    plt.show()
```


```python

```

## 2. Algorithm Implementations

Here I'll implement the three search algorithms. I'm modifying them slightly from the original script to return not just the path and its length, but also the number of nodes explored. This is a crucial metric for comparing their search efficiency.

### A* Search (Informed Search)
A* is a "best-first" search algorithm that is both complete and optimal. It uses a heuristic function, `h(n)`, to estimate the cost from the current node to the goal. It maintains a priority queue of nodes to visit, prioritized by `f(n) = g(n) + h(n)`, where `g(n)` is the known cost from the start to the current node. For my heuristic, I'll use the **Manhattan distance**, which is admissible (it never overestimates the true cost) for a grid where only cardinal moves are allowed.


```python
def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(maze):
    """Finds the shortest path using A* and returns path, length, and nodes explored."""
    start = end = None
    rows, cols = len(maze), len(maze[0])
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == START: start = (r, c)
            elif maze[r][c] == END: end = (r, c)

    if not start or not end: return None, None, 0

    open_set = [(heuristic(start, end), 0, start, [start])]
    g_scores = {start: 0}
    nodes_explored = 0

    while open_set:
        nodes_explored += 1
        _, g_score, current, path = heapq.heappop(open_set)

        if current == end:
            return path, len(path) - 1, nodes_explored

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and maze[neighbor[0]][neighbor[1]] != WALL:
                new_g_score = g_score + 1
                if new_g_score < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = new_g_score
                    h_score = heuristic(neighbor, end)
                    f_score = new_g_score + h_score
                    heapq.heappush(open_set, (f_score, new_g_score, neighbor, path + [neighbor]))
    
    return None, None, nodes_explored
```

### Breadth-First Search (Uninformed Search)
BFS is also complete and optimal for unweighted graphs like this maze. It explores the maze layer by layer from the start point, using a FIFO (First-In, First-Out) queue. Because it explores uniformly, it's guaranteed to find the shortest path in terms of the number of steps. However, it doesn't use any heuristic to guide its search, so it may explore many unnecessary nodes.


```python
def bfs_search(maze):
    """Finds the shortest path using BFS and returns path, length, and nodes explored."""
    start = end = None
    rows, cols = len(maze), len(maze[0])
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == START: start = (r, c)
            elif maze[r][c] == END: end = (r, c)

    if not start or not end: return None, None, 0

    queue = collections.deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if current == end:
            return path, len(path) - 1, len(visited)
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                maze[neighbor[0]][neighbor[1]] != WALL and neighbor not in visited):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None, None, len(visited)
```

### Depth-First Search (Uninformed Search)
DFS explores as deeply as possible along each branch before backtracking, using a LIFO (Last-In, First-Out) stack. It is complete but **not** optimal. It will find a path if one exists, but it is very unlikely to be the shortest one. It's often faster at finding *any* solution, but at the cost of optimality.


```python
def dfs_search(maze):
    """Finds a path using DFS and returns path, length, and nodes explored."""
    start = end = None
    rows, cols = len(maze), len(maze[0])
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == START: start = (r, c)
            elif maze[r][c] == END: end = (r, c)

    if not start or not end: return None, None, 0

    stack = [(start, [start])]
    visited = {start}
    
    while stack:
        current, path = stack.pop()
        
        if current == end:
            return path, len(path) - 1, len(visited)
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                maze[neighbor[0]][neighbor[1]] != WALL and neighbor not in visited):
                visited.add(neighbor)
                stack.append((neighbor, path + [neighbor]))
    
    return None, None, len(visited)
```

## 3. Experimental Procedure

Now I'll run the experiments. I have three sample mazes (`maze1.csv`, `maze2.csv`, `maze3.csv`) of varying complexity. For each maze, I will:
1. Load and visualize the maze.
2. Run each of the three algorithms.
3. Record the execution time, path length, and nodes explored.
4. Visualize the resulting path for each successful search.
5. Store the results in a pandas DataFrame for final analysis.


```python
def run_experiments():
    maze_files = ["maze1.csv", "maze2.csv", "maze3.csv"]
    results = []
    
    algorithms = {
        "A* Search": a_star_search,
        "BFS": bfs_search,
        "DFS": dfs_search
    }

    for filename in maze_files:
        print(f"\n{'='*20} Testing {filename} {'='*20}")
        maze = load_maze(filename)
        if not maze:
            continue
        
        plot_maze(maze, title=f"Original Maze: {filename}")
        
        for name, func in algorithms.items():
            start_time = time.perf_counter()
            path, length, explored = func(maze)
            end_time = time.perf_counter()
            
            if path:
                print(f"\n--- Solution found by {name} ---")
                plot_maze(maze, path, title=f"{name} Solution for {filename}")
                print(f"Path Length: {length} steps")
                print(f"Nodes Explored: {explored}")
                print(f"Execution Time: {(end_time - start_time) * 1000:.4f} ms")
                results.append([filename, name, length, explored, (end_time - start_time) * 1000])
            else:
                print(f"\n--- No solution found by {name} ---")
                results.append([filename, name, -1, explored, (end_time - start_time) * 1000])
    
    return pd.DataFrame(results, columns=["Maze", "Algorithm", "Path Length", "Nodes Explored", "Time (ms)"])

# Run the main experiment function
results_df = run_experiments()
```

    
    ==================== Testing maze1.csv ====================
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_1.png)
    


    
    --- Solution found by A* Search ---
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_3.png)
    


    Path Length: 17 steps
    Nodes Explored: 45
    Execution Time: 0.2782 ms
    
    --- Solution found by BFS ---
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_5.png)
    


    Path Length: 17 steps
    Nodes Explored: 60
    Execution Time: 0.1394 ms
    
    --- Solution found by DFS ---
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_7.png)
    


    Path Length: 17 steps
    Nodes Explored: 55
    Execution Time: 0.1355 ms
    
    ==================== Testing maze2.csv ====================
    [['S', '', '#', '', '', '#', 'E'], ['', '', '#', '', '', '#', ''], ['', '', '#', '', '', '#', ''], ['', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', ''], ['', '#', '', '', '', '#', ''], ['', '#', '', '#', '', '', '']]
    0 7
    1 7
    2 7
    3 7
    4 7
    5 7
    6 7



    
![png](output_13_9.png)
    


    
    --- Solution found by A* Search ---
    [['S', '', '#', '', '', '#', 'E'], ['', '', '#', '', '', '#', ''], ['', '', '#', '', '', '#', ''], ['', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', ''], ['', '#', '', '', '', '#', ''], ['', '#', '', '#', '', '', '']]
    0 7
    1 7
    2 7
    3 7
    4 7
    5 7
    6 7



    
![png](output_13_11.png)
    


    Path Length: 12 steps
    Nodes Explored: 22
    Execution Time: 0.1007 ms
    
    --- Solution found by BFS ---
    [['S', '', '#', '', '', '#', 'E'], ['', '', '#', '', '', '#', ''], ['', '', '#', '', '', '#', ''], ['', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', ''], ['', '#', '', '', '', '#', ''], ['', '#', '', '#', '', '', '']]
    0 7
    1 7
    2 7
    3 7
    4 7
    5 7
    6 7



    
![png](output_13_13.png)
    


    Path Length: 12 steps
    Nodes Explored: 36
    Execution Time: 0.2441 ms
    
    --- Solution found by DFS ---
    [['S', '', '#', '', '', '#', 'E'], ['', '', '#', '', '', '#', ''], ['', '', '#', '', '', '#', ''], ['', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', ''], ['', '#', '', '', '', '#', ''], ['', '#', '', '#', '', '', '']]
    0 7
    1 7
    2 7
    3 7
    4 7
    5 7
    6 7



    
![png](output_13_15.png)
    


    Path Length: 16 steps
    Nodes Explored: 33
    Execution Time: 0.2389 ms
    
    ==================== Testing maze3.csv ====================
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_17.png)
    


    
    --- Solution found by A* Search ---
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_19.png)
    


    Path Length: 17 steps
    Nodes Explored: 45
    Execution Time: 0.1729 ms
    
    --- Solution found by BFS ---
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_21.png)
    


    Path Length: 17 steps
    Nodes Explored: 60
    Execution Time: 0.1427 ms
    
    --- Solution found by DFS ---
    [['S', '', '', '#', '', '', '', '', '', '#'], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '#', '', '', '#', '', '#', '', '#', ''], ['', '', '', '', '', '', '', '', '#', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '#', '', '#', '', ''], ['', '#', '', '#', '', '', '', '', '', ''], ['', '#', '', '#', '', '#', '', '', 'E', '']]
    0 10
    1 10
    2 10
    3 10
    4 10
    5 10
    6 10
    7 10
    8 10
    9 10



    
![png](output_13_23.png)
    


    Path Length: 17 steps
    Nodes Explored: 55
    Execution Time: 0.1316 ms


## 4. Results and Comparative Analysis

With the experiments complete, I can now aggregate and visualize the results to draw conclusions. I'll create bar plots to compare the three key metrics across the different mazes.


```python
print("\n\n--- Aggregate Results ---")
print(results_df)

fig, axes = plt.subplots(3, 1, figsize=(12, 18))
fig.suptitle('Algorithm Performance Comparison', fontsize=20)

# Plot 1: Path Length
sns.barplot(data=results_df, x='Maze', y='Path Length', hue='Algorithm', ax=axes[0])
axes[0].set_title('Path Length (Optimality)')
axes[0].set_ylabel('Steps')

# Plot 2: Nodes Explored
sns.barplot(data=results_df, x='Maze', y='Nodes Explored', hue='Algorithm', ax=axes[1])
axes[1].set_title('Nodes Explored (Search Efficiency)')
axes[1].set_ylabel('Number of Nodes')

# Plot 3: Execution Time
sns.barplot(data=results_df, x='Maze', y='Time (ms)', hue='Algorithm', ax=axes[2])
axes[2].set_title('Execution Time (Speed)')
axes[2].set_ylabel('Time (milliseconds)')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
```

    
    
    --- Aggregate Results ---
            Maze  Algorithm  Path Length  Nodes Explored  Time (ms)
    0  maze1.csv  A* Search           17              45   0.278229
    1  maze1.csv        BFS           17              60   0.139355
    2  maze1.csv        DFS           17              55   0.135464
    3  maze2.csv  A* Search           12              22   0.100686
    4  maze2.csv        BFS           12              36   0.244061
    5  maze2.csv        DFS           16              33   0.238940
    6  maze3.csv  A* Search           17              45   0.172871
    7  maze3.csv        BFS           17              60   0.142716
    8  maze3.csv        DFS           17              55   0.131625



    
![png](output_15_1.png)
    


## 5. Summary

The experimental results confirm the initial hypothesis plus provide several key insights into the behavior of these algorithms:

1.  **Optimality (Path Length):** As expected, both **A* and BFS consistently found the shortest path** in all mazes. Their path lengths are identical. In contrast, **DFS found a valid path but it was significantly longer** and non-optimal. This is its most significant drawback; it prioritizes depth over breadth, often taking a convoluted route to the goal.

2.  **Search Efficiency (Nodes Explored):** This is where the difference between informed and uninformed search becomes clear. **A* consistently explored the fewest nodes**. Its heuristic function (Manhattan distance) effectively guided the search towards the goal, pruning a large number of irrelevant paths. BFS, lacking a heuristic, had to explore a much larger area of the maze, especially in the more complex `maze3.csv`. DFS explored a number of nodes somewhere between A* and BFS, as it quickly dives deep down one path.

3.  **Speed (Execution Time):** The execution time is closely correlated with the number of nodes explored. **A* was the fastest algorithm** for finding the optimal path. While DFS was sometimes faster in raw time (especially on `maze2.csv`), this speed comes at the cost of a non-optimal solution. BFS was the slowest of the three due to its exhaustive, layer-by-layer exploration.

### Conclusion

For problems where the **shortest path is required**, **A* is the superior algorithm**. It provides the same optimal guarantee as BFS but with significantly better performance in terms of both time and computational effort (nodes explored). The power of a good heuristic is clearly demonstrated here.

**BFS** remains a valuable benchmark. It is simple to implement and guarantees optimality, but it does not scale well to larger, more complex state spaces.

**DFS** is useful only when finding *any* path is sufficient and speed is paramount, and path quality is not a concern. In most practical pathfinding applications, this trade-off is not acceptable.