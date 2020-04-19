# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>
# <editor-fold desc="Import RX">
from rx import from_list, range
from rx.operators import map, to_list, zip
# </editor-fold>
# <editor-fold desc="Import Numpy">
from numpy.core.multiarray import ndarray
# </editor-fold>
# <editor-fold desc="Import OpenCV">
import cv2
# </editor-fold>
# <editor-fold desc="Import Other Libraries">
import re
# </editor-fold>

# <editor-fold desc=" Import Own Classes">
from models.Model.SourceStrategy import SourceStrategy
# </editor-fold>


# noinspection PyMethodMayBeStatic
class VideoStubSourceStrategy(SourceStrategy):

    # <editor-fold desc="Constructor">
    def __init__(self):
        super().__init__()

    # </editor-fold>

    # <editor-fold desc="Abstract methods">
    def _get_extension_list(self) -> List[str]:
        return [".mp4", ".avi",
                ".mov", ".mpeg",
                ".flv", ".wmv"]

    def _get_image_list(self, file_name: str) -> List[Tuple[str, ndarray]]:
        video_frame_list = []
        video = cv2.VideoCapture(file_name)
        while True:
            next_available, frame = video.read()
            if not next_available:
                break
            video_frame_list.append(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        video.release()

        image_list: List[Tuple[str, ndarray]] = (
            range(0, len(video_frame_list) - 1)
            .pipe(
                zip(
                    from_list(video_frame_list),
                    range(0, len(video_frame_list)-1)
                )
            )
            .pipe(map(
                lambda zip_element:
                (re.sub("\.\./\.\./data/", "", file_name) + "_Frame_" + str(zip_element[0]), zip_element[1])
            ))
            .pipe(to_list())
            .run()
        )
        return image_list

    # </editor-fold>
