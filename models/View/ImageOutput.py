# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# <editor-fold desc="Import RX">
from rx import from_list
from rx.operators import map, filter, to_list
# </editor-fold>
# <editor-fold desc="Import Numpy">
import numpy
from numpy.core.multiarray import ndarray
# </editor-fold>
# <editor-fold desc="Import cv2">
import cv2
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Controller.Configuration import *
# </editor-fold>
from models.Controller.Vehicle import Vehicle


# noinspection PyMethodMayBeStatic,DuplicatedCode
class ImageOutput:
    # <editor-fold desc="Constructor">
    def __init__(self):
        self._annotated_resized_image: Optional[ndarray] = None
        self._annotated_background_subtract: Optional[ndarray] = None
        self._annotated_close_transformation: Optional[ndarray] = None
        self._annotated_frame_counter: Optional[ndarray] = None
        self._frame_counter: int = 0
        self._vehicle_list: List[Vehicle] = []
    # </editor-fold>

    # <editor-fold desc="Public interface">
    def set_annotated_resized_image(self, annotated_resized_image: Optional[ndarray]):
        self._annotated_resized_image = annotated_resized_image

    def set_annotated_background_subtract(self, annotated_background_subtract: Optional[ndarray]):
        self._annotated_background_subtract = annotated_background_subtract

    def set_annotated_close_transformation(self, annotated_close_transformation: Optional[ndarray]):
        self._annotated_close_transformation = annotated_close_transformation

    def set_vehicle_list(self, vehicle_list: List[Vehicle]):
        self._vehicle_list = vehicle_list

    def annotate(self):
        self._annotate_recognition_frame()
        (
            from_list(self._vehicle_list)
            .subscribe(lambda vehicle: self._annotate_bounding_box(
                vehicle.get_bounding_box_centre_point(),
                vehicle.get_bounding_box_frame_points(),
                vehicle.annotation_colour()
            ))
        )
        self._annotate_frame_counter()
        cv2.imshow('Resized Image', self._annotated_resized_image)
        cv2.imshow('Background Substract', self._annotated_background_subtract)
        cv2.imshow('Close Transformation', self._annotated_close_transformation)
        cv2.imshow('Frame counter', self._annotated_frame_counter)
        self._frame_counter = self._frame_counter + 1
    # </editor-fold>

    # <editor-fold desc="Annotate methods">
    def _is_inside_recognition_frame(self, bounding_box_centre_point: Tuple[int, int]) -> bool:
        correct_horizontal_position = left_border < bounding_box_centre_point[0] < right_border
        correct_vertical_position = top_border < bounding_box_centre_point[1] < bottom_border
        if correct_horizontal_position and correct_vertical_position:
            return True
        else:
            return False

    def _annotate_recognition_frame(self):

        def annotate_lines(cv2_frame: ndarray) -> ndarray:
            frame = cv2.line(cv2_frame, tlc, trc, colour_recognition_frame, thickness_recognition_frame)
            frame = cv2.line(frame, blc, brc, colour_recognition_frame, thickness_recognition_frame)
            frame = cv2.line(frame, tlc, blc, colour_recognition_frame, thickness_recognition_frame)
            frame = cv2.line(frame, trc, brc, colour_recognition_frame, thickness_recognition_frame)
            return frame

        tlc: Tuple[int, int] = (left_border, top_border)
        trc: Tuple[int, int] = (right_border, top_border)
        blc: Tuple[int, int] = (left_border, bottom_border)
        brc: Tuple[int, int] = (right_border, bottom_border)
        annotated_frames = (
            from_list(
                [self._annotated_resized_image,
                 self._annotated_background_subtract,
                 self._annotated_close_transformation]
            )
            .pipe(map(
                lambda cv2_frame: annotate_lines(cv2_frame=cv2_frame)
            ))
            .pipe(to_list())
            .run()
        )
        self._annotated_resized_image = annotated_frames[0]
        self._annotated_background_subtract = annotated_frames[1]
        self._annotated_close_transformation = annotated_frames[2]

    def _annotate_bounding_box(self,
                               bounding_box_centre_point: Tuple[int, int],
                               bounding_box_frame_points: Tuple[Tuple[int, int], Tuple[int, int]],
                               bounding_box_frame_colour: Tuple[int, int, int]):

        def annotate_lines(cv2_frame: ndarray) -> ndarray:
            frame = cv2.circle(
                cv2_frame, bounding_box_centre_point, 2, colour_bounding_box_centre, thickness_bounding_box_centre
            )
            frame = cv2.rectangle(
                frame, bounding_box_frame_points[0],
                bounding_box_frame_points[1], bounding_box_frame_colour,
                thickness_bounding_box_frame
            )
            return frame

        if self._is_inside_recognition_frame(bounding_box_centre_point=bounding_box_centre_point):
            annotated_frames = (
                from_list(
                    [self._annotated_resized_image,
                     self._annotated_background_subtract,
                     self._annotated_close_transformation]
                )
                .pipe(map(
                    lambda cv2_frame:
                    annotate_lines(cv2_frame)
                ))
                .pipe(to_list())
                .run()
            )
            self._annotated_resized_image = annotated_frames[0]
            self._annotated_background_subtract = annotated_frames[1]
            self._annotated_close_transformation = annotated_frames[2]

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
        self._annotated_frame_counter = numpy.zeros((video_resolution[1], video_resolution[0], 3))
        self._annotated_frame_counter = cv2.putText(
            self._annotated_frame_counter,
            "Frames: " + str(self._frame_counter),
            (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2
        )
        self._annotated_frame_counter = cv2.putText(
            self._annotated_frame_counter,
            "Vehicles driving downwards: " + str(vehicles_downwards),
            (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, colour_for_downwards_vehicles, 2
        )
        self._annotated_frame_counter = cv2.putText(
            self._annotated_frame_counter,
            "Vehicles driving upwards: " + str(vehicles_upwards),
            (0, 150), cv2.FONT_HERSHEY_PLAIN, 2, colour_for_upwards_vehicles, 2
        )
        self._annotated_frame_counter = cv2.putText(
            self._annotated_frame_counter,
            "Vehicles driving direction not determined: " + str(vehicles_undecided),
            (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, colour_for_undetermined_vehicles, 2
        )
    # </editor-fold>
