# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# </editor-fold>#
# <editor-fold desc="Import Class Type">
from abc import ABC, abstractmethod
# </editor-fold>
# <editor-fold desc="Import RX">
from rx import from_list
from rx.operators import map, filter, to_list, flat_map
# </editor-fold>
# <editor-fold desc="Import Numpy">
from numpy.core.multiarray import ndarray
# </editor-fold>
# <editor-fold desc="Import Other Libraries">
import os
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Model.Image import Image
# </editor-fold>


class SourceStrategy(ABC):

    # <editor-fold desc="Constructor">
    def __init__(self):
        self._image_store: List[Image] = self._setup_image_store()
        self._image_store_pointer: int = self._setup_image_store_pointer()

    # </editor-fold>

    # <editor-fold desc="Public interface">
    def get_image_store(self) -> List[Image]:
        return self._image_store

    def get_image_store_pointer(self) -> int:
        return self._image_store_pointer

    def return_next_image(self) -> Optional[Image]:
        return (
            None
            if self._image_store_pointer < 0 or self._image_store_pointer > len(self._image_store) - 1
            else self._image_store[self._image_store_pointer]
        )

    def is_next_image(self) -> bool:
        return (
            False
            if self._image_store_pointer < 0 or self._image_store_pointer > len(self._image_store) - 1
            else True
        )

    def reset_image_store_pointer(self):
        self._image_store_pointer = 0

    # </editor-fold>

    # <editor-fold desc="Abstract methods"
    @abstractmethod
    def _get_extension_list(self) -> List[str]:
        pass

    @abstractmethod
    def _get_image_list(self, file_name: str) -> List[Tuple[str, ndarray]]:
        pass

    # </editor-fold>

    # <editor-fold desc="Setup methods">
    def _setup_image_store(self) -> List[Image]:
        def _get_file_names(path) -> List[str]:
            output: List[str] = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    output = output + [os.path.join(root, file)]
            return output

        def _create_image(image_name: str, image_data: ndarray) -> Image:
            image_width: int = image_data.shape[1]
            image_height: int = image_data.shape[0]
            image_bw: bool = len(image_data.shape) <= 2
            return Image(
                image_name=image_name, image_height=image_height,
                image_width=image_width, image_bw=image_bw,
                image_data=image_data
            )

        image_store: List[Image] = (
            from_list(
                self._get_extension_list()
            )
            .pipe(flat_map(
                lambda extension:
                from_list(
                    _get_file_names("../../data")
                )
                .pipe(filter(
                    lambda file_name: file_name.endswith(extension)
                ))
            ))
            .pipe(flat_map(
                lambda file_name: from_list(self._get_image_list(file_name=file_name))
            ))
            .pipe(map(
                lambda file_data: _create_image(image_name=file_data[0], image_data=file_data[1])
            ))
            .pipe(to_list())
            .run()
        )
        return image_store

    def _setup_image_store_pointer(self) -> int:
        if len(self._image_store) > 0:
            return 0
        else:
            return -1

    # </editor-fold>
