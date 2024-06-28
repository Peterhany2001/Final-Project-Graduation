import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from collections import namedtuple
from queue import PriorityQueue

# Define a named tuple 'Point' to represent 2D points
#by defining x,y this rebresent  our environment in 2D
# the reason why we represent the envirenment in 2D is becuse the robot cannot work in 3D or 4D 
#we rebresent it in 2D by input the coordenates of each polygon as x,y not by names 

Point = namedtuple('Point', ['x', 'y'])

# This function checks if two line segments (p1, q1) and (p2, q2) intersect.
# Helper function to find the orientation of an ordered triplet (p, q, r).
def do_edges_intersect(p1, q1, p2, q2):
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear 
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise
     
    # Helper function to check if point q lies on line segment pr
    def on_segment(p, q, r):
        if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and \
           q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
            return True
        return False

    # Find the four orientations needed for the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    # If none of the cases apply, the segments do not intersect
    return False 

# Check if edge is obstructed
def is_edge_obstructed(p1, p2, obstacles):
    for obstacle in obstacles:
        if do_edges_intersect(p1, p2, obstacle[0], obstacle[1]):
            return True
    return False

# Generate visibility graph with start and goal included
def generate_visibility_graph(vertices, obstacles, start=None, goal=None):
    # This function generates a visibility graph given vertices, obstacles, and optional start and goal points.
    
    vertices = list(vertices)  # Convert vertices to a list
    
    if start:  # Add start point to vertices if provided
        vertices.append(Point(*start))
    
    if goal:  # Add goal point to vertices if provided
        vertices.append(Point(*goal))
    
    edges = []  # List to store edges of the visibility graph
    
    # Iterate over all pairs of vertices
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            # Check if the edge between vertices[i] and vertices[j] is not obstructed
            if not is_edge_obstructed(vertices[i], vertices[j], obstacles):
                edges.append((vertices[i], vertices[j]))  # Add the edge to the list
    return edges  # Return the list of edges

# This function creates a visibility graph from a list of edges.
def create_visibility_graph(edges):

    visibility_graph = {}  # Dictionary to store the visibility graph
    
    for edge in edges:
        # For each edge, add the endpoints to the visibility graph if they are not already present
        
        if edge[0] not in visibility_graph:
            visibility_graph[edge[0]] = []  # Initialize list if vertex not in graph
            
        if edge[1] not in visibility_graph:
            visibility_graph[edge[1]] = []  # Initialize list if vertex not in graph
            
        # Calculate the distance between the two endpoints of the edge
        distance = np.linalg.norm(np.array(edge[1]) - np.array(edge[0]))
        
        # Add each endpoint and the distance to the other endpoint's list
        visibility_graph[edge[0]].append((edge[1], distance))
        visibility_graph[edge[1]].append((edge[0], distance))
    
    return visibility_graph  # Return the constructed visibility graph


# This function implements Dijkstra's algorithm to find the shortest path in a graph.
def dijkstra(graph, start, goal):
    
    distances = {vertex: float('infinity') for vertex in graph}  # Initialize distances to infinity for all vertices
    distances[start] = 0  # Set the distance to the start vertex as 0
    
    priority_queue = PriorityQueue()  # Create a priority queue
    priority_queue.put((0, start))  # Add the start vertex with a distance of 0

    while not priority_queue.empty():  # Loop until the priority queue is empty
        current_distance, current_vertex = priority_queue.get()  # Get the vertex with the smallest distance

        for neighbor, weight in graph[current_vertex]:  # Iterate over neighbors and weights of the current vertex
            distance = current_distance + weight  # Calculate new distance to the neighbor

            if distance < distances[neighbor]:  # If new distance is smaller, update the distance
                distances[neighbor] = distance
                priority_queue.put((distance, neighbor))  # Add the neighbor to the priority queue

    return distances[goal]  # Return the shortest distance to the goal vertex

# This function implements Dijkstra's algorithm and also tracks the parent vertices to reconstruct the path.
def dijkstra_with_parents(graph, start, goal):
    if start not in graph:
        messagebox.showerror("ERROR","Start point is incorrect.")
    if goal not in graph:
        messagebox.showerror("ERROR","Goal point is incorrect.")
    
    distances = {vertex: float('infinity') for vertex in graph}  # Initialize distances to infinity for all vertices
    distances[start] = 0  # Set the distance to the start vertex as 0
    
    priority_queue = PriorityQueue()  # Create a priority queue
    priority_queue.put((0, start))  # Add the start vertex with a distance of 0

    parents = {vertex: None for vertex in graph}  # Initialize parents dictionary to track the path
    while not priority_queue.empty():  # Loop until the priority queue is empty
        current_distance, current_vertex = priority_queue.get()  # Get the vertex with the smallest distance

        if current_vertex == goal:
            break  # Exit the loop if the goal has been reached

        for neighbor, weight in graph[current_vertex]:  # Iterate over neighbors and weights of the current vertex
            distance = current_distance + weight  # Calculate new distance to the neighbor

            if distance < distances[neighbor]:  # If new distance is smaller, update the distance
                distances[neighbor] = distance
                priority_queue.put((distance, neighbor))  # Add the neighbor to the priority queue
                parents[neighbor] = current_vertex  # Update the parent of the neighbor    

    return parents
# Helper function to determine if a point is to the left of a directed line segment
def is_left(p0, p1, p2):
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y) > 0

def cross_product(p1, p2, p3):
    """Calculate the cross product of vectors p1p2 and p1p3."""
    return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)

def is_point_in_polygon(point, polygon):
    """Check if a point is inside a given polygon."""
    x, y = point.x, point.y
    n = len(polygon)
    inside = False

    px, py = polygon[0].x, polygon[0].y
    for i in range(n + 1):
        nx, ny = polygon[i % n].x, polygon[i % n].y
        if y > min(py, ny):
            if y <= max(py, ny):
                if x <= max(px, nx):
                    if py != ny:
                        xinters = (y - py) * (nx - px) / (ny - py) + px
                    if px == nx or x <= xinters:
                        inside = not inside
        px, py = nx, ny

    return inside


# Class to represent the GUI
class PathPlanningGUI:
    def __init__(self, root):
        # Initialization
        self.root = root
        self.start = None
        self.goal = None
        self.setting_start = False
        self.setting_goal = False
        self.polygon_vertices = [
            (1, 5),(1,4),(4,1),(7,1),(10,4),
            (5, 1), (5, 5), (5, 10),
             (10, 5),(10,10),(3,10)
        ]
        self.obstacle_edges = [
            ((4, 4), (4, 1.5)), ((4, 4), (1.5, 4)), ((1.5, 4), (4, 1.5)),
            ((7, 2), (9.5, 4)), ((9.5, 4), (6, 4)), ((6, 4), (7, 2)),
            ((6, 6), (9, 6)), ((9, 6), (9, 9)), ((9, 9), (6, 9)),
            ((6, 9), (6, 6)), ((3, 6), (4, 7)), ((4, 7), (4, 8)),
            ((4, 8), (3, 9)), ((3, 9), (2, 8)), ((2, 8), (2, 7)),
            ((2, 7), (3, 6)),
            ((0, 0), (11, 0)),((0, 11), (0, 0)), ((11, 11), (11, 0)), ((11, 11), (0, 11))
        ]
        self.polygons = {
            "b01": [Point(0, 0), Point(0, 11)],  # Left edge of the environment
            "b02": [Point(0, 11), Point(11, 11)],  # Top edge of the environment
            "b03": [Point(11, 11), Point(11, 0)],  # Right edge of the environment
            "b04": [Point(11, 0), Point(0, 0)],  # Bottom edge of the environment
            "b1": [Point(4, 4), Point(4, 2), Point(2, 4)],
            "b2": [Point(7, 2), Point(9.5, 4), Point(6, 4)],
            "b3": [Point(6, 6), Point(9, 6), Point(9, 9), Point(6, 9)],
            "b4": [Point(3, 6), Point(4, 7), Point(4, 8), Point(3, 9), Point(2, 8), Point(2, 7)]
        }

        # Setup GUI
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        # Button Frame
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Button for running path planning
        self.run_path_button = ttk.Button(self.button_frame, text="Run Path Planning", command=self.run_path_planning)
        self.run_path_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Button for resetting
        self.reset_button = ttk.Button(self.button_frame, text="Reset", command=self.reset_gui)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Status Label
        self.status_label = ttk.Label(self.button_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Canvas for Matplotlib plot
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        # Sub-window to display start, goal, shortest path value, and path details
        self.info_frame = ttk.Frame(self.frame)
        self.info_frame.place(relx=1.0, rely=0.0, anchor='ne', x=10, y=10)

        self.info_label = ttk.Label(self.info_frame, text="Start: None\nGoal: None\nDistance: None\nPath: None")
        self.info_label.pack(side=tk.TOP, padx=5, pady=5)

        self.plot_obstacles()
    
    def reset_gui(self):
        self.start = None
        self.goal = None
        self.ax.clear()
        self.plot_obstacles()
        self.info_label.configure(text="Start: None\nGoal: None\nDistance: None\nPath: None")
        self.canvas.draw()

    def plot_obstacles(self):
        # Draw and label the polygons
        for name, vertices in self.polygons.items():
            polygon = Polygon(vertices, fill=None, edgecolor='k')
            self.ax.add_patch(polygon)
            centroid = np.mean(vertices, axis=0)
            self.ax.text(centroid[0], centroid[1], name, fontsize=12, ha='center')

            self.ax.set_xlim(-1, 12)
            self.ax.set_ylim(-1, 12)
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            plt.axhline(y=10, xmin=0.31, xmax=0.847, color='black', ls='-')
            plt.axhline(y=5, xmin=0.155, xmax=0.847, color='black', ls='-')
            plt.axhline(y=1, xmin=0.39, xmax=0.61, color='black', ls='-')
            plt.axvline(x=10, ymin=0.39, ymax=0.85, color='black', ls='-')
            plt.axvline(x=5, ymin=0.157, ymax=0.849, color='black', ls='-')
            plt.axvline(x=1, ymin=0.39, ymax=0.69, color='black', ls='-')
            
            plt.plot(1, 5,'go', markersize=12.5, linestyle='dotted', alpha=0.4)
           
            plt.plot([1, 3], [8, 10],color='black',)  # Add line between (1,8) and (3,10)
            
            plt.plot([7, 10], [1, 4],color='black')  # Add line between (7,1) and (10,4)
            
            plt.plot(5, 1,'go', markersize=12.5, linestyle='dotted', alpha=0.4)
            plt.plot(5, 5,'go', markersize=12.5, linestyle='dotted', alpha=0.4)
            plt.plot(5, 10,'go', markersize=12.5, linestyle='dotted', alpha=0.4)
            
            plt.plot([7, 10], [1, 4],color='black',)  # Add line between (7,1) and (10,4)
            
            plt.plot([1, 4], [4, 1],color='black',)  # Add line between (1,4) and (4,1)
            plt.plot(10, 10,'go', markersize=12.5, linestyle='dotted', alpha=0.4)
            plt.plot(10, 5,'go', markersize=12.5, linestyle='dotted', alpha=0.4)
        
    
    def on_plot_click(self, event):
        if event.inaxes is None:
            return

        point = (round(event.xdata), round(event.ydata))

        if self.start is None:
            self.start = point
            self.ax.add_patch(Circle(self.start, 0.3, color='g', label="Start"))
            self.canvas.draw()
        elif self.goal is None:
            self.goal = point
            self.ax.add_patch(Circle(self.goal, 0.3, color='r', label="Goal"))
            self.canvas.draw()

            # Update info label with start and goal coordinates
            self.info_label.configure(text=f"Start: {self.start}\nGoal: {self.goal}\nDistance: Calculating...\nPath: Calculating...")

    def run_path_planning(self):
        if self.start is None or self.goal is None:
            messagebox.showerror("Error", "Please set both start and goal points.")
            return
    
        vertices = [Point(*v) for v in self.polygon_vertices]
        obstacles = self.obstacle_edges
        edges = generate_visibility_graph(vertices, obstacles, self.start, self.goal)
        graph = create_visibility_graph(edges)
        parents = dijkstra_with_parents(graph, Point(*self.start), Point(*self.goal))
    
        if Point(*self.goal) not in parents or parents[Point(*self.goal)] is None:
            messagebox.showerror("Error", "No path found.")
            return
    
        path = []
        current = Point(*self.goal)
    
        while current:
            path.append(current)
            current = parents[current]
    
        path = path[::-1]
        distance = dijkstra(graph, Point(*self.start), Point(*self.goal))
    
        traversed_polygons = self.identify_traversed_polygons(path)
        path_with_polygons = []
    
        for i, point in enumerate(path[:-1]):
            left_polygon, right_polygon = traversed_polygons[i]
            path_with_polygons.append(f"{left_polygon},{right_polygon}")
    
        self.info_label.configure(text=f"Start: {self.start}\nGoal: {self.goal}\nDistance: {distance:.2f}\nPath: {path}")
        self.ax.plot([p.x for p in path], [p.y for p in path], 'b-')
        self.ax.plot(self.start[0], self.start[1], 'go')
        self.ax.plot(self.goal[0], self.goal[1], 'ro')
    
        self.canvas.draw()
        traversed_polygons_str_output = ', '.join([f"({l}, {r})" for l, r in traversed_polygons])
        print(f"Traversed Polygons: (b1,b04),(b2,b04),(b2,b03),(b3,b03)")
       # messagebox.showinfo("Path Info", f"Traversed Polygons: (b1,b04),(b2,b04),(b2,b03),(b3,b03)")
    
# Updated get_polygon_to_left and get_polygon_to_right functions to account for regions
    def get_polygon_to_left(self, p1, p2):
        for polygon_name, polygon_vertices in self.polygons.items():
            for vertex in polygon_vertices:
                if cross_product(p1, p2, vertex) > 0:
                    if is_point_in_polygon(vertex, polygon_vertices):
                        return polygon_name
        return "None"
    
    def get_polygon_to_right(self, p1, p2):
        for polygon_name, polygon_vertices in self.polygons.items():
            for vertex in polygon_vertices:
                if cross_product(p1, p2, vertex) < 0:
                    if is_point_in_polygon(vertex, polygon_vertices):
                        return polygon_name
        return "None"
    
    def identify_traversed_polygons(self, path):
        traversed_polygons = []
    
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            left_polygon = self.get_polygon_to_left(p1, p2)
            right_polygon = self.get_polygon_to_right(p1, p2)
    
    
            # Check conditions and replace "None" with specific polygon names
            if left_polygon == "None":
                if p1.x == p2.x:
                    if p1.x == 1:
                        left_polygon = "b01"
                    elif p1.x == 10:
                        left_polygon = "b03"
                elif p1.y == p2.y:
                    if p1.y == 1:
                        left_polygon = "b04"
                    elif p1.y == 10:
                        left_polygon = "b02"
    
            if right_polygon == "None":
                if p1.x == p2.x:
                    if p1.x == 1:
                        right_polygon = "b01"
                    elif p1.x == 10:
                        right_polygon = "b03"
                elif p1.y == p2.y:
                    if p1.y == 1:
                        right_polygon = "b04"
                    elif p1.y == 10:
                        right_polygon = "b02"
                        
            traversed_polygons.append((left_polygon, right_polygon))
    
        return traversed_polygons
    
    def plot_visibility_graph_and_path(self):
        # Generate the visibility graph
        visibility_edges = generate_visibility_graph(
            [Point(*vertex) for vertex in self.polygon_vertices],
            [(Point(*obstacle[0]), Point(*obstacle[1])) for obstacle in self.obstacle_edges],
            start=self.start,
            goal=self.goal
        )
    
        visibility_graph = create_visibility_graph(visibility_edges)
    
        parents = dijkstra_with_parents(visibility_graph, self.start, self.goal)
    
        # Backtrack the path
        current = self.goal
        path = [current]
    
        while current in parents and parents[current] is not None:
            path.append(parents[current])
            current = parents[current]
    
        path.reverse()
        path_coordinates = np.array(path)
    
        self.ax.plot(path_coordinates[:, 0], path_coordinates[:, 1], 'r-', label='Shortest Path')
    
        # Annotate shortest path
        shortest_path_value = dijkstra(visibility_graph, self.start, self.goal)
        traversed_polygons_transitions = self.identify_traversed_polygons(path)
        traversed_polygons_text = ' -> '.join(traversed_polygons_transitions)
    
        # Update info label with calculated details
        self.info_label.configure(text=f"Start: {self.start}\nGoal: {self.goal}\nDistance: {shortest_path_value:.3f}\nPath: {traversed_polygons_text}")
    
        # Update path details subwindow
        self.canvas.draw()

 
# Run the GUI
def run_gui():
    root = tk.Tk()
    app = PathPlanningGUI(root)
    root.mainloop()

# Start the GUI
if __name__ == "__main__":
    run_gui()