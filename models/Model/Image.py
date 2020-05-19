# <editor-fold desc="Import Numpy">
from numpy.core.multiarray import ndarray
# </editor-fold>


class Image:

    # <editor-fold desc="Constructor">
    def __init__(self, image_name: str, image_height: int, image_width: int, image_bw: bool, image_data: ndarray):
        self._image_name: str = image_name
        self._image_height: int = image_height
        self._image_width: int = image_width
        self._image_bw: bool = image_bw
        self._image_data: ndarray = image_data

    # </editor-fold>

    # <editor-fold desc="Public interface">
    def get_image_name(self) -> str:
        return self._image_name

    def get_image_height(self) -> int:
        return self._image_height

    def get_image_widt(self) -> int:
        return self._image_width

    def get_image_bw(self) -> bool:
        return self._image_bw

    def get_image_data(self) -> ndarray:
        return self._image_data

    # </editor-fold>
