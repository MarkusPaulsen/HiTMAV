from typing import *
# <editor-fold desc="Import RX">
from rx import from_list
from rx.operators import map, filter, flat_map, first
# </editor-fold>

import cv2
import numpy as np
from numpy.core.multiarray import ndarray

from models.Model.Image import Image

# Implemented according to https://github.com/AlfaCodeFlow/Vehicle_Detection-And-Classification


# noinspection PyMethodMayBeStatic
# noinspection PyShadowingNames
class VehicleDetection:

    def __init__(self):
        self._recent_resized_image: Optional[ndarray] = None
        self._recent_background_subtractor: Optional[ndarray] = None
        self._recent_transformation: Optional[ndarray] = None
        self.background_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2()  # 200, 90, False)

    def apply_background_subtractor(self, resized_image: ndarray):
        output = self.background_subtractor_mog2.apply(resized_image)
        self._recent_background_subtractor = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def apply_transformation(self, transformation: ndarray):
        output = cv2.morphologyEx(transformation, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        self._recent_transformation = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        return output

    def create_contours(self, resized_image: ndarray) -> List[ndarray]:
        (image, contours, hierarchy) = (
            from_list([resized_image])
            .pipe(map(
                lambda resized_image:
                self.apply_background_subtractor(resized_image=resized_image)
            ))
            .pipe(map(
                lambda foreground_mask:
                cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
            ))
            .pipe(map(
                lambda threshold_structure:
                cv2.morphologyEx(threshold_structure[1], cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            ))
            .pipe(map(
                lambda transformation:
                self.apply_transformation(transformation=transformation)
            ))
            .pipe(map(
                lambda transformation:
                cv2.findContours(transformation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            ))
            .pipe(first())
            .run()
        )
        return contours

    def get_bounding_box_centre(self, contour: ndarray) -> Tuple[int, int]:
        bounding_box_centre = (
            from_list([contour])
            .pipe(map(
                lambda contour:
                cv2.moments(contour)
            ))
            .pipe(map(
                lambda moments:
                (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            ))
            .pipe(first())
            .run()
        )
        return bounding_box_centre

    def get_bounding_box(self, contour: ndarray) -> Tuple[int, int, int, int]:
        return cv2.boundingRect(contour)

    def annotate_bounding_box(self, contour_data: Tuple[Tuple[int, int], Tuple[int, int, int, int]]):
        bounding_box_centre_x_y: Tuple[int, int] = contour_data[0]
        bounding_box_x_y: Tuple[int, int] = (contour_data[1][0], contour_data[1][1])
        bounding_box_x_y_2: Tuple[int, int] = (
            contour_data[1][0] + contour_data[1][2], contour_data[1][1] + contour_data[1][3]
        )
        self._recent_resized_image = cv2.circle(
            self._recent_resized_image, bounding_box_centre_x_y, 2, (0, 0, 255), -1
        )
        self._recent_resized_image = cv2.rectangle(
            self._recent_resized_image, bounding_box_x_y, bounding_box_x_y_2, (0, 255, 0), 2
        )
        self._recent_background_subtractor = cv2.circle(
            self._recent_background_subtractor, bounding_box_centre_x_y, 2, (0, 0, 255), -1
        )
        self._recent_background_subtractor = cv2.rectangle(
            self._recent_background_subtractor, bounding_box_x_y, bounding_box_x_y_2, (0, 255, 0), 2
        )
        self._recent_transformation = cv2.circle(
            self._recent_transformation, bounding_box_centre_x_y, 2, (0, 0, 255), -1
        )
        self._recent_transformation = cv2.rectangle(
            self._recent_transformation, bounding_box_x_y, bounding_box_x_y_2, (0, 255, 0), 2
        )

    def annotate_image(self, image: Image):
        self._recent_resized_image = cv2.resize(image.get_image_data(), (900, 500))
        (
            from_list([self._recent_resized_image])
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
                (self.get_bounding_box_centre(contour=contour), self.get_bounding_box(contour=contour))
            ))
            .subscribe(
                lambda contour_data:
                self.annotate_bounding_box(contour_data=contour_data)
            )
        )
        cv2.imshow('Resized Image', self._recent_resized_image)
        cv2.imshow('Background Substractor', self._recent_background_subtractor)
        cv2.imshow('Transformation', self._recent_transformation)
