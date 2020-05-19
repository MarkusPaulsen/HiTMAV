# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# <editor-fold desc="Import RX">
from rx import from_list
from rx.operators import filter, to_list
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Controller.Vehicle import Vehicle
# </editor-fold>


# noinspection PyShadowingNames
class VehicleManager:
    # <editor-fold desc="Constructor">
    def __init__(self):
        self._vehicle_list: List[Vehicle] = []
    # </editor-fold>

    # <editor-fold desc="Public interface">
    def get_vehicle_list(self) -> List[Vehicle]:
        return self._vehicle_list

    def update_vehicles(
            self, bounding_box_centre_points: List[Tuple[int, int]],
            bounding_box_frame_points: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ):
        self._delete_outdated_vehicles()
        self._update_vehicles_positions(bounding_box_centre_points, bounding_box_frame_points)
    # </editor-fold>

    # <editor-fold desc="Adjust vehicles methods">
    def _update_vehicles_positions(self,
                                   bounding_box_centre_points: List[Tuple[int, int]],
                                   bounding_box_frame_points: List[Tuple[Tuple[int, int], Tuple[int, int]]]):

        def create_vehicles_to_update_list(bounding_box_centre_point: Tuple[int, int]) -> List[Vehicle]:
            vehicles_to_update: List[Vehicle] = (
                from_list(self._vehicle_list)
                .pipe(filter(
                    lambda vehicle:
                    vehicle.updateable(bounding_box_centre_point)
                ))
                .pipe(to_list())
                .run()
            )
            return vehicles_to_update
        for point in zip(bounding_box_centre_points, bounding_box_frame_points):
            vehicles_to_update: List[Vehicle] = create_vehicles_to_update_list(point[0])
            if not vehicles_to_update:
                self._vehicle_list.append(
                    Vehicle(
                        bounding_box_centre_point=point[0],
                        bounding_box_frame_points=point[1]
                    )
                )
            else:
                vehicles_to_update.sort(key=(
                    lambda vehicle:
                    vehicle.difference(bounding_box_centre_point=point[0])
                ))
                vehicles_to_update[0].update_position(
                    bounding_box_centre_point=point[0],
                    bounding_box_frame_points=point[1]
                )

    def _delete_outdated_vehicles(self):
        self._vehicle_list = (
            from_list(self._vehicle_list)
            .pipe(filter(
                lambda vehicle:
                vehicle.to_keep()
            ))
            .pipe(to_list())
            .run()
        )
    # </editor-fold>
