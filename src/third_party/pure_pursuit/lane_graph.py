import math
import carla
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from third_party.common.graphs import DenseGraph
from third_party.pure_pursuit.path_finder import PathFinder
from third_party.common.utils import angular_between_two_vector


class LaneGraph:
    def __init__(
            self,
            dense_graph: DenseGraph
    ):
        """
        LaneGraph class
        Args:
            dense_graph (DenseGraph): dense graph
        """
        self.map = map

        # init
        self.__dense_graph = dense_graph
        self.graph = self.__dense_graph.graph

        self.path_finder = PathFinder(dense_graph)

    def _get_loc(
            self,
            index: int
    ) -> carla.Location:
        """
        Get location from node index
        Args:
            index (int): index of node in dense graph

        Returns:
            (carla.Location): location of node
        """
        node_instance = self.__dense_graph.nodes[index]
        node_loc = (node_instance.transform.location
                    if isinstance(node_instance, carla.Waypoint)
                    else node_instance)
        return node_loc

    @staticmethod
    def _dist(
            node: Union[carla.Waypoint, carla.Location],
            loc: carla.Location,
            is_waypoint: bool = True
    ) -> float:
        """
        Calculate the distance from node to loc
        Args:
            node (Union[carla.Waypoint, carla.Location]): node
            loc (carla.Location): location
            is_waypoint (bool): whether node is carla.Waypoint or not (carla.Location)

        Returns:
            (float): distance from node to loc
        """
        if is_waypoint:
            return node.transform.location.distance(loc)
        else:
            return node.distance(loc)

    def create_temporal_link(
            self,
            transform: carla.Transform,
            d_nearest_node: float = 10.,
            alpha: float = np.pi / 8
    ) -> tuple:
        """
        Create connection between vehicle node and target_node.
        Target_node is a node that has smallest distance to vehicle_node
        and must be in FOV
        Args:
            transform (carla.Transform): vehicle transform
            d_nearest_node (float): maximum distance to nearest node
            alpha (float) FOV = 2*alpha

        Returns:
            (tuple):
            (
                vehicle_node (int): index node of vehicle,
                is_new_vehicle_node (bool): whether is vehicle node a new node
                target_node (int): index node of target node
                is_new_target_node (bool): whether is nearest node a new node
            )

        """
        location = transform.location
        veh_node, is_new_vehicle_node = self.__dense_graph.get_index_node(location)
        if not is_new_vehicle_node:
            return veh_node, False, -1, False

        def get_target_node() -> tuple:
            """
            Get target node to link vehicle_node to target_node
            The target node is the closest node (C) to vehicle node
            and in FOV (2*alpha)

            Returns:
                (tuple):
                (
                    (int): target node
                    (bool): is new nearest node
                )
            """
            # find nearest node to vehicle node
            node_dist = sorted(
                [
                    (
                        idx,
                        self._dist(
                            node=node,
                            loc=location,
                            is_waypoint=isinstance(node, carla.Waypoint)
                        )
                    ) for idx, node in self.__dense_graph.nodes.items()
                    if idx != veh_node
                ],
                key=lambda x: x[1]
            )

            # suppose vector heading only for 2d
            v_heading = carla.Vector3D(
                math.cos(transform.rotation.yaw),
                math.sin(transform.rotation.yaw),
                0
            )

            # scan to get node which is in FOV (2*alpha)
            # and has distance to vehicle node <= self.d_nearest_node
            for nearest_node, d_node in node_dist:
                # break if distance to vehicle node > self.d_nearest_node
                if d_node > d_nearest_node:
                    break

                node_loc = self._get_loc(nearest_node)
                # vector vehicle_node
                v_veh_node = carla.Location(node_loc - location)

                angular = angular_between_two_vector(
                    u=np.array([v_heading.x, v_heading.y, v_heading.z]),
                    v=np.array([v_veh_node.x, v_veh_node.y, v_veh_node.z])
                )

                # only get node in FOV
                if np.abs(angular) > alpha:
                    continue

                # and this node must not be a dead-end (have neighbours)
                num_neighbours = len(self.__dense_graph.graph[nearest_node])
                if num_neighbours == 0:
                    continue

                # and could create a desirable path (have small angular)
                first_neighbour = [k for k in self.__dense_graph.graph[nearest_node].keys()][0]
                first_neighbour_loc = self._get_loc(first_neighbour)
                v_node_neighbour = carla.Location(first_neighbour_loc - node_loc)

                n_angular = angular_between_two_vector(
                    u=np.array([v_veh_node.x, v_veh_node.y, v_veh_node.z]),
                    v=np.array([v_node_neighbour.x, v_node_neighbour.y, v_node_neighbour.z])
                )
                if n_angular > alpha:
                    continue

                return nearest_node, False

            # if there is no suitable node
            # -> create new temp node as nearest node
            # temp node is a location which is forward to current vehicle location
            # and far "d_nearest_node" away
            next_vector = location + v_heading * d_nearest_node
            nearest_node_loc = carla.Location(next_vector)
            # create new node
            nearest_node, _ = self.__dense_graph.get_index_node(nearest_node_loc)
            return nearest_node, True

        # find target node
        # link vehicle node to target node
        target_node, is_new_target_node = get_target_node()
        dist = self._dist(
            node=self.__dense_graph.nodes[target_node],
            loc=location,
            is_waypoint=isinstance(self.__dense_graph.nodes[target_node], carla.Waypoint)
        )

        # add edge
        self.__dense_graph.add_edge_by_node(veh_node, target_node, dist)

        return (
            veh_node, is_new_vehicle_node,
            target_node, is_new_target_node
        )

    def plot_graph(
            self,
            paths: list = (),
    ) -> None:
        """
        Visualize graph and paths (if len(paths) > 0)
        Args:
            paths (list): list of paths
        Returns:

        """
        if len(self.__dense_graph.nodes) > 0:
            # plot graph
            pos = np.array([_pos for _pos in self.__dense_graph.pos_by_node.values()])
            x, y, _ = pos.T
            plt.scatter(x, y, color="blue", s=1, label="waypoint")

        # plot path
        for i, path in enumerate(paths):
            plt.plot(
                path[:, 0], path[:, 1],
                linewidth=3, color="orange", label="path" if i == 0 else ""
            )

    def get_paths(
            self,
            transform: carla.Transform,
            distance: float = 50.,
    ) -> list:
        """
        Get possible reference paths
        Args:
            transform (carla.Transform): current transform of vehicle
            distance (float): distance of each path

        Returns:
            (list(np.ndarray)): list of possible reference paths given a position
        """
        # create temporal link in graph
        (veh_node, is_new_veh_node,
         trg_node, is_new_trg_node) = self.create_temporal_link(transform=transform)

        # # FOR DEBUGGING
        # target_loc = self.__dense_graph.nodes[trg_node]
        # plt.scatter(target_loc.x, target_loc.y, marker='o', c='r')

        # find paths
        paths, _ = self.path_finder.bfs(start_node=veh_node, max_dist=distance)

        # get path in position (x, y, z)
        paths = self.__dense_graph.convert_path_node_to_pos(paths)

        # remove temporal nodes
        if is_new_veh_node:
            self.__dense_graph.remove_node(veh_node)
        if is_new_trg_node:
            self.__dense_graph.remove_node(trg_node)

        return paths
