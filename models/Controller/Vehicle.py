# Class inspired (and code partitially taken) from an implementation by the GitHub user AlfaCodeFlow:
# https://github.com/AlfaCodeFlow/Vehicle_Detection-And-Classification

# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# <editor-fold desc="Import Other Libraries">
import math
import time
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Controller.Configuration import *
# </editor-fold>


class Vehicle:

    # <editor-fold desc="Constructor">
    def __init__(self,
                 bounding_box_centre_point: Tuple[int, int],
                 bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]]):
        self._is_driving_downwards: Optional[bool] = None
        self._current_position: Tuple[int, int] = bounding_box_centre_point
        self._last_position: Optional[Tuple[int, int]] = None
        self._pre_last_position: Optional[Tuple[int, int]] = None
        self._bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]] = bounding_box_frame_points
        self._time_stamp: float = time.time()
    # </editor-fold>

    # <editor-fold desc="Public Interface">
    def get_is_driving_downwards(self) -> Optional[bool]:
        return self._is_driving_downwards

    def get_bounding_box_centre_point(self) -> Tuple[int, int]:
        return self._current_position

    def get_bounding_box_frame_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return self._bounding_box_frame_points

    def difference(self, bounding_box_centre_point: Tuple[int, int]):
        difference_vector: Tuple[int, int] = (
            bounding_box_centre_point[0] - self._current_position[0],
            bounding_box_centre_point[1] - self._current_position[1]
        )
        difference_vector_length = math.sqrt(pow(difference_vector[0], 2) + pow(difference_vector[1], 2))
        return difference_vector_length

    def updateable(self, bounding_box_centre_point: Tuple[int, int]) -> bool:
        if self.difference(bounding_box_centre_point=bounding_box_centre_point) < 20:
            return True
        else:
            return False

    def update_position(self,
                        bounding_box_centre_point: Tuple[int, int],
                        bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]]):
        self._pre_last_position = self._last_position
        self._last_position = self._current_position
        self._current_position = bounding_box_centre_point

        if self._pre_last_position is not None\
                and self._last_position is not None\
                and self._current_position is not None:
            if self._pre_last_position[1] < self._last_position[1] < self._current_position[1]:
                self._is_driving_downwards = True
            elif self._pre_last_position[1] > self._last_position[1] > self._current_position[1]:
                self._is_driving_downwards = False
            else:
                pass

        self._bounding_box_frame_points = bounding_box_frame_points
        self._time_stamp = time.time()

    def annotation_colour(self):
        if self._is_driving_downwards is not None:
            if self._is_driving_downwards:
                return colour_for_downwards_vehicles
            else:
                return colour_for_upwards_vehicles
        else:
            return colour_for_undetermined_vehicles

    def to_keep(self):
        if time.time() - self._time_stamp >= deletion_time:
            return False
        else:
            return True
    # </editor-fold>
