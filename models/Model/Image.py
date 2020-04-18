class Image:
    def __init__(self, filename: str, height: int, width: int, bw: bool, data: bytes):
        self.filename: str = filename
        self.height: int = height
        self.width: int = width
        self.bw: bool = bw
        self.data: bytes = data
