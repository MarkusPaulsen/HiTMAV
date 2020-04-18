# <editor-fold desc="Import Class Type">
from abc import ABC, abstractmethod
# </editor-fold>

from models.Model.Image import Image


class Source(ABC):
    @abstractmethod
    def get_next_image(self) -> Image:
        pass
