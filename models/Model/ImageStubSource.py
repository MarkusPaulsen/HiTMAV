from models.Model.Image import Image
from models.Model.Source import Source


class ImageStubSource(Source):

    def __init__(self):
        super().__init__()

    def get_next_image(self) -> Image:
        pass
