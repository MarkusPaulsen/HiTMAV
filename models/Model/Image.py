from numpy.core.multiarray import ndarray


class Image:
    def __init__(self, image_name: str, image_height: int, image_width: int, image_bw: bool, image_data: ndarray):
        self.image_name: str = image_name
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.image_bw: bool = image_bw
        self.image_data: ndarray = image_data
