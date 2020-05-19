# <editor-fold desc="Import Typing">
from typing import *
# </editor-fold>

# <editor-fold desc="Import Own Classes">
from models.Model.SourceStrategy import SourceStrategy
from models.Model.ImageSourceStrategy import ImageStubSourceStrategy
from models.Model.VideoSourceStrategy import VideoSourceStrategy
from models.Model.SourceLoaderPolicy import SourceLoaderPolicy
# </editor-fold>


# noinspection PyMethodMayBeStatic
class SourceLoaderContext:

    # <editor-fold desc="Constructor">
    def __init__(self):
        self._source_loader_policy: SourceLoaderPolicy = self._setup_source_loader_policy()
        self._source_strategy: Optional[SourceStrategy] = self._setup_source_strategy()

    # </editor-fold>

    # <editor-fold desc="Public interface">
    def get_source_strategy(self) -> Optional[SourceStrategy]:
        return self._source_strategy

    def get_source_loader_policy(self) -> SourceLoaderPolicy:
        return self._source_loader_policy

    def update_source_strategy(self):
        self._source_strategy = self._setup_source_strategy()

    # </editor-fold>

    # <editor-fold desc="Setup methods">
    def _setup_source_strategy(self) -> Optional[SourceStrategy]:
        self._source_loader_policy.update_best_strategy()
        best_strategy: str = self._source_loader_policy.get_best_strategy()
        if best_strategy == "Image":
            return ImageStubSourceStrategy()
        elif best_strategy == "Video":
            return VideoSourceStrategy()
        else:
            return None

    def _setup_source_loader_policy(self) -> SourceLoaderPolicy:
        return SourceLoaderPolicy()

    # </editor-fold>
