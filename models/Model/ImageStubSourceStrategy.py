# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# <editor-fold desc="Import Numpy">
from numpy.core.multiarray import ndarray
# </editor-fold>
# <editor-fold desc="Import OpenCV">
import cv2
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Model.SourceStrategy import SourceStrategy
# </editor-fold>


class ImageStubSourceStrategy(SourceStrategy):

    # <editor-fold desc="Constructor">
    def __init__(self):
        super().__init__()

    # </editor-fold>

    # <editor-fold desc="Abstract methods">
    def _get_extension_list(self) -> List[str]:
        return [".jpg", ".jpeg",
                ".jpe", ".jp2",
                ".bmp", ".dib",
                ".png", ".ppm",
                ".pbm", ".pgm",
                ".tif", ".tiff",
                ".sr", ".ras"]

    def _get_image_list(self, file_name: str) -> List[Tuple[str, ndarray]]:
        image_data: ndarray = cv2.imread(file_name)
        return [(file_name, image_data)]

    # </editor-fold>
