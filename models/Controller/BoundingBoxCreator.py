# <editor-fold desc="Import Typing">
from typing import *
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
from models.Model.Configuration import *
# </editor-fold>


class ContoursCreator:

    # <editor-fold desc="Constructor">
    def __init__(self, image: ndarray):
        self._image = image
        self._resized_image: Optional[ndarray] = None
        self._background_subtractor_mog2: BackgroundSubtractorMOG2 = self._setup_background_subtractor_mog2()
        self._background_subtract: Optional[ndarray] = self._setup_background_subtract()
        self._threshold: Optional[Tuple[float, ndarray]] = self._setup_threshold()
        self._open_transformation: Optional[ndarray] = self._setup_open_transformation()
        self._close_transformation: Optional[ndarray] = self._setup_close_transformation()
        self._contours: List[ndarray] = self._setup_contours()
    # </editor-fold>

    # <editor-fold desc="Public interface">
    def get_contours(self) -> List[ndarray]:
        return self._contours
    # </editor-fold>

    # <editor-fold desc="Setup methods">
    def _setup_background_subtractor_mog2(self) -> BackgroundSubtractorMOG2:
        return cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    def _setup_resized_image(self) -> ndarray:
        return cv2.resize(self._image, video_resolution)
    
    def _setup_background_subtract(self) -> ndarray:
        return self._background_subtractor_mog2.apply(self._resized_image)

    def _setup_threshold(self) -> Tuple[float, ndarray]:
        return cv2.threshold(self._background_subtract, 200, 255, cv2.THRESH_BINARY)

    def _setup_open_transformation(self) -> ndarray:
        return cv2.morphologyEx(self._threshold[1], cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))

    def _setup_close_transformation(self) -> ndarray:
        return cv2.morphologyEx(self._open_transformation, cv2.MORPH_CLOSE, numpy.ones((11, 11), numpy.uint8))

    def _setup_contours(self) -> List[ndarray]:
        (image, contours, hierarchy) = cv2.findContours(
            self._close_transformation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        return contours
    # </editor-fold>
