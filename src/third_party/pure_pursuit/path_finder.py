class PathFinder:
    def __init__(
            self,
            dense_graph
    ):
        self.__dense_graph = dense_graph
        self.graph = dense_graph.graph

        self.visited = None
        self.traverse = None
        self.distance = None

    def __init(self):
        self.visited = dict([(node, False) for node in self.__dense_graph.nodes.keys()])
        self.traverse = dict()
        self.distance = dict()

    def bfs(self, start_node, max_dist):
        """
        BFS algorithm to find all path from start_node
        which have distance >= max_dist and nearest to max_dist
        Args:
            start_node (int): index of starting node
            max_dist (float): max distance of paths

        Returns:
            (tuple): (
                list of paths,
                distance of each path
            )
        """
        self.__init()

        queue = list()
        queue.append(start_node)
        self.visited[start_node] = True
        self.distance[start_node] = 0.

        while queue:
            current_node = queue.pop(0)
            if self.distance[current_node] >= max_dist:
                continue

            for neighbour, dist in self.graph[current_node].items():
                if not self.visited[neighbour]:
                    queue.append(neighbour)
                    self.traverse[neighbour] = current_node
                    self.distance[neighbour] = self.distance[current_node] + self.graph[current_node][neighbour]
                    self.visited[neighbour] = True

        # sort self.distance descending
        self.distance = dict(
            sorted(
                [(k, d) for k, d in self.distance.items()],
                key=lambda x: x[1], reverse=True
            )
        )

        # get result
        list_of_path = list()
        list_of_dist = list()
        for end_node, dist in self.distance.items():
            # get the longest path only
            existed = False
            for path in list_of_path:
                if end_node in path:
                    existed = True
                    break
            if existed:
                continue

            # get trajectory
            path = [end_node]
            while end_node != start_node:
                end_node = self.traverse[end_node]
                path.append(end_node)

            # traceback
            path = [p for p in reversed(path)]

            list_of_path.append(path)
            list_of_dist.append(dist)

        return (
            list_of_path,
            list_of_dist
        )
