# noinspection PyMethodMayBeStatic
class SourceLoaderPolicy:

    # <editor-fold desc="Constructor">
    def __init__(self):
        self._best_strategy: str = self._setup_best_strategy()

    # </editor-fold>

    # <editor-fold desc="Public interface">
    def get_best_strategy(self) -> str:
        return self._best_strategy

    def update_best_strategy(self):
        self._best_strategy = "Image Stub"

    # </editor-fold>

    # <editor-fold desc="Setup methods">
    def _setup_best_strategy(self) -> str:
        return "Image Stub"

    # </editor-fold>
