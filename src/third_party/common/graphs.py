import carla
import numpy as np
from typing import Union
from collections import defaultdict

DECIMALS = 2


class DirectedGraph:
    def __init__(self):
        """
        Weighted directed graph
        """
        self.graph = defaultdict(defaultdict)
        self.nodes = dict()
        self.pos_by_node = dict()
        self.node_by_pos = dict()

    def convert_path_node_to_pos(
            self,
            paths: list
    ):
        """
        Convert index node of paths to its position on map
        Args:
            paths (list): list of paths in index node
                Example: paths = [ [0, 2, 10], [0, 1, 3, 5, 9], ... ]

        Returns:

        """
        paths_in_position = list()
        for path in paths:
            pos = np.array([self.pos_by_node[node] for node in path])
            paths_in_position.append(pos)
        return paths_in_position

    def get_index_node(
            self,
            data: Union[carla.Waypoint, carla.Location],
            decimals: int = DECIMALS
    ) -> tuple:
        """
        Get index of node
        if this node is already in graph -> return its index
        else create new node
        Args:
            data (carla.WayPoint, carla.Location): waypoint or location
            decimals (float): decimal to round node's position,
                purpose of this work is to stack nearby nodes into one (if too close)

        Returns:
            (tuple): (
                node_index (int): index of node, return None if input is not Waypoint or Location
                is_new_node (bool): is this node a new node in graph
            )
        """

        def get_position():
            if isinstance(data, carla.Waypoint):
                return (
                    data.transform.location.x,
                    data.transform.location.y,
                    data.transform.location.z
                )
            elif isinstance(data, carla.Location):
                return (
                    data.x,
                    data.y,
                    data.z
                )
            else:
                return None

        pos = get_position()
        if pos is None:
            return None, False

        x, y, z = pos
        round_x = float(np.round(x, decimals=decimals))
        round_y = float(np.round(y, decimals=decimals))
        round_z = float(np.round(z, decimals=decimals))
        key_pos = (round_x, round_y, round_z)

        # create new node
        if key_pos not in self.node_by_pos:
            # for quick access
            new_idx = len(self.node_by_pos)
            self.node_by_pos[key_pos] = new_idx
            self.pos_by_node[new_idx] = key_pos

            # add node
            self.nodes[new_idx] = data
            return new_idx, True
        else:
            return self.node_by_pos[key_pos], False

    def add_edge_by_waypoint(
            self,
            w1: carla.Waypoint,
            w2: carla.Waypoint
    ) -> tuple:
        """
        Add edge to graph using waypoint
        Args:
            w1 (carla.Waypoint): first waypoint
            w2 (carla.Waypoint): second waypoint

        Returns:
            (tuple): (
                new_w1 (bool): is first waypoint a new node in graph
                new_w2 (bool): is second waypoint a new node in graph
            )
        """
        idx_pts_1, new_w1 = self.get_index_node(w1)
        idx_pts_2, new_w2 = self.get_index_node(w2)

        w2_loc = w2.transform.location
        dist = w1.transform.location.distance(w2_loc)

        self.add_edge_by_node(
            u=idx_pts_1,
            v=idx_pts_2,
            weight=dist
        )
        return new_w1, new_w2

    def add_edge_by_location(
            self,
            l1: carla.Location,
            l2: carla.Location
    ) -> tuple:
        """
        Add edge to graph using location
        Args:
            l1 (carla.Location): first location
            l2 (carla.Location): second location

        Returns:
            (tuple): (
                new_l1 (bool): is first location a new node in graph
                new_l2 (bool): is second location a new node in graph
            )
        """
        idx_pts_1, new_l1 = self.get_index_node(l1)
        idx_pts_2, new_l2 = self.get_index_node(l2)
        dist = l1.distance(l2)

        self.add_edge_by_node(
            u=idx_pts_1,
            v=idx_pts_2,
            weight=dist
        )
        return new_l1, new_l2

    def add_edge_by_node(
            self,
            u: int,
            v: int,
            weight: float
    ) -> None:
        """
        Add edge to graph
        (u and v are already in graph)
        Args:
            u (int): index of start node
            v (int): index of edn node
            weight (float): weight of edge
        Returns:
            None
        """
        self.graph[u][v] = weight

    def remove_node(
            self,
            index: int
    ):
        """
        Remove a node in graph
        Args:
            index (int): index of a node

        Returns:

        """
        if index in self.nodes:
            del self.nodes[index]
        if index in self.graph:
            del self.graph[index]
        if index in self.pos_by_node:
            pos = self.pos_by_node[index]
            del self.pos_by_node[index]
            del self.node_by_pos[pos]

        for n, neighbours in self.graph.items():
            for neighbour in neighbours.keys():
                if neighbour == index:
                    del self.graph[n][neighbour]

    def remove_edge(
            self,
            u: int,
            v: int
    ):
        """
        Remove a edge between 2 nodes in graph
        Args:
            u (int) start node
            v (int) end node

        Returns:

        """
        if u in self.graph:
            if v in self.graph[u]:
                del self.graph[u][v]


class DenseGraph(DirectedGraph):
    def __init__(
            self,
            topology: list,
            sampling_distance: float
    ):
        """
        This class will create a dense graph from
            + map.get_topology()
            + list of 2 carla.Location value
        Distance of 2 consecutive points of graph is sampling_distance
        Args:
            topology (list(tuple)):
                + list of tuple (w1, w2), show the directed vector from w1 to w2,
                with w1, w2 are carla.Waypoint
                + list of tuple (l1, l2) where l1, l2 is carla.Location
            sampling_distance (float): distance of 2 consecutive points
        """
        super().__init__()
        self.topology = topology
        self.sampling_distance = sampling_distance
        self.create_graph()

    def create_graph(
            self,
    ):
        """
        Create graph from map topology
        Graph is dense-graph and will be stored in self.graph
        Distance of 2 consecutive points
        Returns:

        """
        # create empty graph if len(self.topology) == 0
        if len(self.topology) == 0:
            return

        # for waypoint instance
        if isinstance(self.topology[0][0], carla.Waypoint):

            for w1, w2 in self.topology:
                end_loc = w2.transform.location
                d_w1_w2 = w1.transform.location.distance(end_loc)

                if w1.transform.location.distance(end_loc) > self.sampling_distance:
                    curr_w = w1
                    next_w = curr_w.next(self.sampling_distance)[0]
                    while True:
                        # create edge (curr_w, next_w)
                        _, is_new_next_w = self.add_edge_by_waypoint(curr_w, next_w)

                        d_next_w2 = next_w.transform.location.distance(end_loc)
                        d_w1_next = w1.transform.location.distance(next_w.transform.location)

                        # stopping condition
                        # create the last edge
                        if d_next_w2 <= self.sampling_distance or d_w1_w2 <= d_w1_next:
                            # create edge (next_w, w2)
                            self.add_edge_by_waypoint(next_w, w2)
                            break

                        # next iter
                        curr_w = next_w
                        next_w = curr_w.next(self.sampling_distance)[0]
                else:
                    # create edge (w1, w2)
                    self.add_edge_by_waypoint(w1, w2)

        # for location instance
        elif isinstance(self.topology[0][0], carla.Location):

            for l1, l2 in self.topology:
                x1, y1 = l1.x, l1.y
                x2, y2 = l2.x, l2.y
                # estimate distance (l1, l2)
                d_l1_l2 = l1.distance(l2)

                # interpolate points between l1 and l2
                num_pts = int(d_l1_l2 / self.sampling_distance)
                interp_x = np.linspace(x1, x2, num_pts)
                interp_y = np.linspace(y1, y2, num_pts)

                for i in range(len(interp_x) - 1):
                    interp_loc1 = carla.Location(x=interp_x[i], y=interp_y[i], z=0)
                    interp_loc2 = carla.Location(x=interp_x[i + 1], y=interp_y[i + 1], z=0)
                    self.add_edge_by_location(interp_loc1, interp_loc2)
