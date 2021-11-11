import json
import math
import pandas as pd


class MapHelper:
    def __init__(self, map_path):
        self._columns = [
            "id",
            "type",
            "x",
            "y",
            "status"
        ]
        self.distance = 50.
        self._get_map(map_path)
        self._get_center_lane_polyline()

    def _get_map(
            self,
            map_path,
    ):
        df = pd.read_csv(map_path)
        self._map = df
        # get lane only
        self._map = self._map.loc[
            (self._map["type"] == "l_lane") |
            (self._map["type"] == "r_lane")
            ]

    def _get_center_lane_polyline(self):
        # deep copy all components in map
        self._center_polyline = self._map.copy(deep=True)

        # group by id and type, and get mean position
        self._center_polyline = self._map.groupby(
            by=self._columns[:2], as_index=False
        ).mean()

    @staticmethod
    def __estimate_distance(
            row,
            agent_x,
            agent_y
    ):
        row["distance"] = math.sqrt(
            (row["x"] - agent_x) ** 2 +
            (row["y"] - agent_y) ** 2
        )
        return row

    def _get_id_lane_in_range(
            self,
            agent_x: float,
            agent_y: float,
    ):
        tmp_center_polyline = self._center_polyline.copy(deep=True)
        tmp_center_polyline = tmp_center_polyline.apply(
            self.__estimate_distance,
            axis=1,  # apply for each row
            agent_x=agent_x,
            agent_y=agent_y
        )
        # get polyline that have distance to agent <= "distance"
        tmp_center_polyline = tmp_center_polyline.loc[
            tmp_center_polyline["distance"] <= self.distance]
        # get ids in list
        ids = tmp_center_polyline[self._columns[0]].to_numpy().tolist()
        return ids

    def _get_lane_by_id(self, list_ids):
        tmp_map = self._map.copy(deep=True)
        tmp_map = tmp_map.loc[tmp_map["id"].isin(list_ids)]

        # group by id, type
        lane_by_id = tmp_map.groupby(by=self._columns[:2])

        # reserve into list container
        lanes = []
        for id_type, frame in lane_by_id:
            # only get numpy of (x, y)
            data = frame[self._columns[2:4]].to_numpy()
            lanes.append(data)
        return lanes

    def get_local_lanes(
            self,
            agent_x: float,
            agent_y: float,
    ):
        # get all lane_id in range distance
        ids = self._get_id_lane_in_range(agent_x, agent_y)
        # get lane polyline
        lanes = self._get_lane_by_id(ids)

        return ids, lanes

    def get_status(self, lane_id):
        status = self._map.loc[lane_id, "status"]
        return json.loads(status)
