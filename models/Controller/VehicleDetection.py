# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# <editor-fold desc="Import RX">
from rx import from_list, just
from rx.operators import map, filter, flat_map, to_list
# </editor-fold>
# <editor-fold desc="Import Numpy">
import numpy
from numpy.core.multiarray import ndarray
# </editor-fold>
# <editor-fold desc="Import cv2">
import cv2
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Model.Image import Image
from models.Model.Configuration import *
# </editor-fold>

# Implemented according to https://github.com/AlfaCodeFlow/Vehicle_Detection-And-Classification


# noinspection PyMethodMayBeStatic
# noinspection PyShadowingNames
from models.Model.Vehicle import Vehicle


# noinspection PyShadowingNames,PyMethodMayBeStatic
class VehicleDetection:

    # <editor-fold desc="Constructor">
    def __init__(self):

        self._background_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._recent_resized_image: Optional[ndarray] = None
        self._recent_background_subtractor: Optional[ndarray] = None
        self._recent_threshold: Optional[Tuple[float, ndarray]] = None
        self._recent_open_transformation: Optional[ndarray] = None
        self._recent_close_transformation: Optional[ndarray] = None
        self._frame_counter_window: Optional[ndarray] = None
        self._vehicle_list: List[Vehicle] = []
        self._frame_counter: int = 0
    # </editor-fold>

    # <editor-fold desc="Contour methods">
    def _apply_background_subtractor(self, resized_image: ndarray) -> ndarray:
        output: ndarray = self._background_subtractor_mog2.apply(resized_image)
        self._recent_background_subtractor = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def _apply_threshold(self, foreground_mask: ndarray) -> Tuple[float, ndarray]:
        output: Tuple[float, ndarray] = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
        self._recent_threshold = cv2.cvtColor(output[1], cv2.COLOR_GRAY2RGB)
        return output

    def _apply_open_transformation(self, threshold_structure: Tuple[float, ndarray]) -> ndarray:
        output: ndarray = cv2.morphologyEx(threshold_structure[1], cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
        self._recent_open_transformation = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def _apply_close_transformation(self, transformation: ndarray) -> ndarray:
        output: ndarray = cv2.morphologyEx(transformation, cv2.MORPH_CLOSE, numpy.ones((11, 11), numpy.uint8))
        self._recent_close_transformation = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def _create_contours(self, resized_image: ndarray) -> List[ndarray]:
        (image, contours, hierarchy) = (
            just(resized_image)
            .pipe(map(
                lambda resized_image:
                self._apply_background_subtractor(resized_image=resized_image)
            ))
            .pipe(map(
                lambda foreground_mask:
                self._apply_threshold(foreground_mask=foreground_mask)
            ))
            .pipe(map(
                lambda threshold_structure:
                self._apply_open_transformation(threshold_structure=threshold_structure)
            ))
            .pipe(map(
                lambda transformation:
                self._apply_close_transformation(transformation=transformation)
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
    def _is_inside_recognition_frame(self, bounding_box_centre_x_y: Tuple[int, int]) -> bool:
        correct_horizontal_position = left_border < bounding_box_centre_x_y[0] < right_border
        correct_vertical_position = top_border < bounding_box_centre_x_y[1] < bottom_border
        if correct_horizontal_position and correct_vertical_position:
            return True
        else:
            return False

    def _annotate_recognition_frame(self):
        tlc: Tuple[int, int] = (left_border, top_border)
        trc: Tuple[int, int] = (right_border, top_border)
        blc: Tuple[int, int] = (left_border, bottom_border)
        brc: Tuple[int, int] = (right_border, bottom_border)
        annotated_frames = (
            from_list(
                [self._recent_resized_image,
                 self._recent_background_subtractor,
                 self._recent_close_transformation]
            )
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, tlc, trc, colour_recognition_frame, thickness_recognition_frame)
            ))
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, blc, brc, colour_recognition_frame, thickness_recognition_frame)
            ))
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, tlc, blc, colour_recognition_frame, thickness_recognition_frame)
            ))
            .pipe(map(
                lambda cv2_frame: cv2.line(cv2_frame, trc, brc, colour_recognition_frame, thickness_recognition_frame)
            ))
            .pipe(to_list())
            .run()
        )
        self._recent_resized_image = annotated_frames[0]
        self._recent_background_subtractor = annotated_frames[1]
        self._recent_close_transformation = annotated_frames[2]
    # </editor-fold>

    # <editor-fold desc="Bounding box methods">
    def _get_bounding_box_centre_point(self, contour: ndarray) -> Tuple[int, int]:
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

    def _get_bounding_box_frame_points(self, contour: ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]] = (
            just(cv2.boundingRect(contour))
            .pipe(map(
                lambda rectangle:
                ((rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]))
            ))
            .run()
        )
        return bounding_box_frame_points

    def _annotate_bounding_box(self,
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
                cv2.circle(
                    cv2_frame, bounding_box_centre_point,
                    2, colour_bounding_box_centre, thickness_bounding_box_centre)
            ))
            .pipe(map(
                lambda cv2_frame:
                cv2.rectangle(
                    cv2_frame, bounding_box_frame_points[0],
                    bounding_box_frame_points[1], bounding_box_frame_colour,
                    thickness_bounding_box_frame
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
    def _create_vehicles_to_update_list(self, bounding_box_centre_point: Tuple[int, int]):
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

    def _update_vehicle_position(self,
                                 bounding_box_centre_point: Tuple[int, int],
                                 bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]]):
        vehicles_to_update: List[Vehicle] = self._create_vehicles_to_update_list(bounding_box_centre_point)
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

    # <editor-fold desc="Frame counter methods">
    def _annotate_frame_counter(self):
        vehicles_undecided = len((
            from_list(self._vehicle_list)
            .pipe(filter(
                lambda vehicle:
                vehicle.get_is_driving_downwards() is None
                and self._is_inside_recognition_frame(vehicle.get_bounding_box_centre_point())
            ))
            .pipe(to_list())
            .run()
        ))
        vehicles_downwards = len((
            from_list(self._vehicle_list)
            .pipe(filter(
                lambda vehicle:
                vehicle.get_is_driving_downwards() is True
                and self._is_inside_recognition_frame(vehicle.get_bounding_box_centre_point())
            ))
            .pipe(to_list())
            .run()
        ))
        vehicles_upwards = len((
            from_list(self._vehicle_list)
            .pipe(filter(
                lambda vehicle:
                vehicle.get_is_driving_downwards() is False
                and self._is_inside_recognition_frame(vehicle.get_bounding_box_centre_point())
            ))
            .pipe(to_list())
            .run()
        ))
        self._frame_counter_window = numpy.zeros((video_resolution[1], video_resolution[0], 3))
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Frames: " + str(self._frame_counter),
            (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2
        )
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Vehicles driving downwards: " + str(vehicles_downwards),
            (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, colour_for_downwards_vehicles, 2
        )
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Vehicles driving upwards: " + str(vehicles_upwards),
            (0, 150), cv2.FONT_HERSHEY_PLAIN, 2, colour_for_upwards_vehicles, 2
        )
        self._frame_counter_window = cv2.putText(
            self._frame_counter_window,
            "Vehicles driving direction not determined: " + str(vehicles_undecided),
            (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, colour_for_undetermined_vehicles, 2
        )
    # </editor-fold>

    # <editor-fold desc="Public interface">
    def annotate_image(self, image: Image):
        self._recent_resized_image = cv2.resize(image.get_image_data(), video_resolution)
        bounding_box_data_list: List[Tuple[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]]] = (
            just(self._recent_resized_image)
            .pipe(flat_map(
                lambda resized_image:
                self._create_contours(resized_image=resized_image)
            ))
            .pipe(filter(
                lambda contour:
                cv2.contourArea(contour) >= vehicle_detection_size
            ))
            .pipe(map(
                lambda contour:
                (
                    self._get_bounding_box_centre_point(contour=contour),
                    self._get_bounding_box_frame_points(contour=contour)
                )
            ))
            .pipe(to_list())
            .run()
        )
        self._annotate_recognition_frame()
        for bounding_box_data in bounding_box_data_list:
            self._update_vehicle_position(bounding_box_data[0], bounding_box_data[1])
            self._delete_outdated_vehicles()
        (
            from_list(self._vehicle_list)
            .subscribe(
                lambda vehicle:
                self._annotate_bounding_box(
                    vehicle.get_bounding_box_centre_point(),
                    vehicle.get_bounding_box_frame_points(),
                    vehicle.annotation_colour()
                ) if self._is_inside_recognition_frame(vehicle.get_bounding_box_centre_point()) else None
            )
        )
        self._annotate_frame_counter()
        cv2.imshow('Resized Image', self._recent_resized_image)
        cv2.imshow('Background Substractor', self._recent_background_subtractor)
        cv2.imshow('Transformation', self._recent_close_transformation)
        cv2.imshow('Frame counter', self._frame_counter_window)
        self._frame_counter = self._frame_counter + 1
    # </editor-fold>
