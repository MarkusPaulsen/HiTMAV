from typing import *
# <editor-fold desc="Import RX">
import numpy
from rx import from_list, just
from rx.operators import map, filter, flat_map, to_list
# </editor-fold>

import cv2
import numpy as np
from numpy.core.multiarray import ndarray

from models.Model.Image import Image

# Implemented according to https://github.com/AlfaCodeFlow/Vehicle_Detection-And-Classification


# noinspection PyMethodMayBeStatic
# noinspection PyShadowingNames
from models.Model.Vehicle import Vehicle


# noinspection PyShadowingNames,PyMethodMayBeStatic
class VehicleDetection:

    def __init__(self):
        self._top_border = 200
        self._bottom_border = 400
        self._left_border = 100
        self._right_border = 700
        self._recent_resized_image: Optional[ndarray] = None
        self._recent_background_subtractor: Optional[ndarray] = None
        self._recent_threshold: Optional[Tuple[float, ndarray]] = None
        self._recent_open_transformation: Optional[ndarray] = None
        self._recent_close_transformation: Optional[ndarray] = None
        self._frame_counter_window: Optional[ndarray] = None
        self.background_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._vehicle_list: List[Vehicle] = []
        self._frame_counter: int = 0

    # <editor-fold desc="Contour methods">
    def apply_background_subtractor(self, resized_image: ndarray) -> ndarray:
        output: ndarray = self.background_subtractor_mog2.apply(resized_image)
        self._recent_background_subtractor = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def apply_threshold(self, foreground_mask: ndarray) -> Tuple[float, ndarray]:
        output: Tuple[float, ndarray] = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
        self._recent_threshold = cv2.cvtColor(output[1], cv2.COLOR_GRAY2RGB)
        return output

    def apply_open_transformation(self, threshold_structure: Tuple[float, ndarray]) -> ndarray:
        output: ndarray = cv2.morphologyEx(threshold_structure[1], cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self._recent_open_transformation = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def apply_close_transformation(self, transformation: ndarray) -> ndarray:
        output: ndarray = cv2.morphologyEx(transformation, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        self._recent_close_transformation = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def create_contours(self, resized_image: ndarray) -> List[ndarray]:
        (image, contours, hierarchy) = (
            just(resized_image)
            .pipe(map(
                lambda resized_image:
                self.apply_background_subtractor(resized_image=resized_image)
            ))
            .pipe(map(
                lambda foreground_mask:
                self.apply_threshold(foreground_mask=foreground_mask)
            ))
            .pipe(map(
                lambda threshold_structure:
                self.apply_open_transformation(threshold_structure=threshold_structure)
            ))
            .pipe(map(
                lambda transformation:
                self.apply_close_transformation(transformation=transformation)
            ))
            .pipe(map(
                lambda transformation:
                cv2.findContours(transformation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ))
            .run()
        )
        return contours
    # </editor-fold>

    # <editor-fold desc="Recognition frame methods">
    def is_inside_recognition_frame(self, bounding_box_centre_x_y: Tuple[int, int]) -> bool:
        correct_horizontal_position = self._left_border < bounding_box_centre_x_y[0] < self._right_border
        correct_vertical_position = self._top_border < bounding_box_centre_x_y[1] < self._bottom_border
        if correct_horizontal_position and correct_vertical_position:
            return True
        else:
            return False

    def annotate_recognition_frame(self):
        tlc: Tuple[int, int] = (self._left_border, self._top_border)
        trc: Tuple[int, int] = (self._right_border, self._top_border)
        blc: Tuple[int, int] = (self._left_border, self._bottom_border)
        brc: Tuple[int, int] = (self._right_border, self._bottom_border)
        annotated_frames = (
            from_list(
                [self._recent_resized_image,
                 self._recent_background_subtractor,
                 self._recent_close_transformation]
            )
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, tlc, trc, (255, 0, 0), 1)
            ))
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, blc, brc, (255, 0, 0), 1)
            ))
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, tlc, blc, (255, 0, 0), 1)
            ))
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, trc, brc, (255, 0, 0), 1)
            ))
            .pipe(to_list())
            .run()
        )
        self._recent_resized_image = annotated_frames[0]
        self._recent_background_subtractor = annotated_frames[1]
        self._recent_close_transformation = annotated_frames[2]
    # </editor-fold>

    # <editor-fold desc="Bounding box methods">
    def get_bounding_box_centre_point(self, contour: ndarray) -> Tuple[int, int]:
        bounding_box_centre_point = (
            just(contour)
            .pipe(map(
                lambda contour:
                cv2.moments(contour)
            ))
            .pipe(map(
                lambda moments:
                (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            ))
            .run()
        )
        return bounding_box_centre_point

    def get_bounding_box_frame_points(self, contour: ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]] = (
            just(cv2.boundingRect(contour))
            .pipe(map(
                lambda rectangle:
                ((rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]))
            ))
            .run()
        )
        return bounding_box_frame_points

    def annotate_bounding_box(self,
                              bounding_box_centre_point: Tuple[int, int],
                              bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]],
                              bounding_box_frame_colour: Tuple[int, int, int]):
        annotated_frames = (
            from_list(
                [self._recent_resized_image,
                 self._recent_background_subtractor,
                 self._recent_close_transformation]
            )
            .pipe(map(
                lambda cv2_frame:
                cv2.circle(cv2_frame, bounding_box_centre_point, 2, (0, 0, 255), -1)
            ))
            .pipe(map(
                lambda cv2_frame:
                cv2.rectangle(
                    cv2_frame, bounding_box_frame_points[0], bounding_box_frame_points[1], bounding_box_frame_colour, 2
                )
            ))
            .pipe(to_list())
            .run()
        )
        self._recent_resized_image = annotated_frames[0]
        self._recent_background_subtractor = annotated_frames[1]
        self._recent_close_transformation = annotated_frames[2]
    # </editor-fold>

    # <editor-fold desc="Vehicle methods">
    def create_vehicles_to_update_list(self, bounding_box_centre_point: Tuple[int, int]):
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

    def update_vehicle_position(self,
                                bounding_box_centre_point: Tuple[int, int],
                                bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]]):
        vehicles_to_update: List[Vehicle] = self.create_vehicles_to_update_list(bounding_box_centre_point)
        if not vehicles_to_update:
            self._vehicle_list.append(
                Vehicle(
                    bounding_box_centre_point=bounding_box_centre_point,
                    bounding_box_frame_points=bounding_box_frame_points
                )
            )
        else:
            vehicles_to_update.sort(key=(
                lambda vehicle:
                vehicle.difference(bounding_box_centre_point=bounding_box_centre_point)
            ))
            vehicles_to_update[0].update_position(
                bounding_box_centre_point=bounding_box_centre_point,
                bounding_box_frame_points=bounding_box_frame_points
            )

    def delete_outdated_vehicles(self):
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

    def annotate_frame_counter(self):
        vehicles_undecided = len((
            from_list(self._vehicle_list)
            .pipe(filter(
                lambda vehicle:
                vehicle.get_is_driving_downwards() is None
                and self.is_inside_recognition_frame(vehicle.get_bounding_box_centre_point())
            ))
            .pipe(to_list())
            .run()
        ))
        vehicles_downwards = len((
            from_list(self._vehicle_list)
            .pipe(filter(
                lambda vehicle:
                vehicle.get_is_driving_downwards() is True
                and self.is_inside_recognition_frame(vehicle.get_bounding_box_centre_point())
            ))
            .pipe(to_list())
            .run()
        ))
        vehicles_upwards = len((
            from_list(self._vehicle_list)
            .pipe(filter(
                lambda vehicle:
                vehicle.get_is_driving_downwards() is False
                and self.is_inside_recognition_frame(vehicle.get_bounding_box_centre_point())
            ))
            .pipe(to_list())
            .run()
        ))
        self._frame_counter_window = numpy.zeros((450, 800, 3))
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Frames: " + str(self._frame_counter),
            (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2
        )
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Vehicles driving downwards: " + str(vehicles_downwards),
            (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2
        )
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Vehicles driving upwards: " + str(vehicles_upwards),
            (0, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2
        )
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Vehicles driving direction not determined: " + str(vehicles_undecided),
            (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2
        )

    def annotate_image(self, image: Image):
        self._recent_resized_image = cv2.resize(image.get_image_data(), (800, 450))
        bounding_box_data_list: List[Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]] = (
            just(self._recent_resized_image)
            .pipe(flat_map(
                lambda resized_image:
                self.create_contours(resized_image=resized_image)
            ))
            .pipe(filter(
                lambda contour:
                cv2.contourArea(contour) > 300
            ))
            .pipe(map(
                lambda contour:
                (
                    self.get_bounding_box_centre_point(contour=contour),
                    self.get_bounding_box_frame_points(contour=contour)
                )
            ))
            .pipe(to_list())
            .run()
        )
        self.annotate_recognition_frame()
        for bounding_box_data in bounding_box_data_list:
            self.update_vehicle_position(bounding_box_data[0], bounding_box_data[1])
            self.delete_outdated_vehicles()
        (
            from_list(self._vehicle_list)
            .subscribe(
                lambda vehicle:
                self.annotate_bounding_box(
                    vehicle.get_bounding_box_centre_point(),
                    vehicle.get_bounding_box_frame_points(),
                    vehicle.annotation_colour()
                ) if self.is_inside_recognition_frame(vehicle.get_bounding_box_centre_point()) else None
            )
        )
        self.annotate_frame_counter()
        cv2.imshow('Resized Image', self._recent_resized_image)
        cv2.imshow('Background Substractor', self._recent_background_subtractor)
        cv2.imshow('Transformation', self._recent_close_transformation)
        cv2.imshow('Frame counter', self._frame_counter_window)
        self._frame_counter = self._frame_counter + 1
