# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# <editor-fold desc="Import RX">
from rx import from_list
from rx.operators import map, to_list
# </editor-fold>
# <editor-fold desc="Import Numpy">
import numpy
from numpy.core.multiarray import ndarray
# </editor-fold>
# <editor-fold desc="Import cv2">
import cv2
from cv2 import BackgroundSubtractorMOG2
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Controller.Configuration import *
# </editor-fold>


# noinspection PyMethodMayBeStatic
class BoundingBoxCreator:

    # <editor-fold desc="Constructor">
    def __init__(self):
        self._background_subtractor_mog2: BackgroundSubtractorMOG2 = \
            cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._resized_image: Optional[ndarray] = None
        self._background_subtract: Optional[ndarray] = None
        self._threshold: Optional[Tuple[float, ndarray]] = None
        self._open_transformation: Optional[ndarray] = None
        self._close_transformation: Optional[ndarray] = None
        self._contours: List[ndarray] = []
        self._bounding_box_centre_points: List[Tuple[int, int]] = []
        self._bounding_box_frame_points: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    # </editor-fold>

    # <editor-fold desc="Public interface">
    def get_resized_image(self) -> ndarray:
        return self._resized_image

    def get_background_subtract(self) -> ndarray:
        return cv2.cvtColor(self._background_subtract, cv2.COLOR_GRAY2RGB)

    def get_close_transformation(self) -> ndarray:
        return cv2.cvtColor(self._close_transformation, cv2.COLOR_GRAY2RGB)

    def get_bounding_box_centre_points(self) -> List[Tuple[int, int]]:
        return self._bounding_box_centre_points

    def get_bounding_box_frame_points(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return self._bounding_box_frame_points
    
    def compute_next_image(self, image: ndarray):
        self._resized_image = cv2.resize(image, video_resolution)
        self._background_subtract = self._background_subtractor_mog2.apply(self._resized_image)
        self._threshold = cv2.threshold(self._background_subtract, 200, 255, cv2.THRESH_BINARY)
        self._open_transformation = cv2.morphologyEx(
            self._threshold[1], cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8)
        )
        self._close_transformation = cv2.morphologyEx(
            self._open_transformation, cv2.MORPH_CLOSE, numpy.ones((11, 11), numpy.uint8)
        )
        (image, contours, hierarchy) = cv2.findContours(
            self._close_transformation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        self._contours = contours
        self._bounding_box_centre_points: List[Tuple[int, int]] = (
            from_list(self._contours)
            .pipe(map(
                lambda contour:
                cv2.moments(contour)
            ))
            .pipe(map(
                lambda moments:
                (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            ))
            .pipe(to_list())
            .run()
        )
        self._bounding_box_frame_points: List[Tuple[Tuple[int, int], Tuple[int, int]]] = (
            from_list(self._contours)
            .pipe(map(
                lambda contour:
                cv2.boundingRect(contour)
            ))
            .pipe(map(
                lambda rectangle:
                ((rectangle[0], rectangle[1]), (rectangle[0] + rectangle[2], rectangle[1] + rectangle[3]))
            ))
            .pipe(to_list())
            .run()
        )
    # </editor-fold>
