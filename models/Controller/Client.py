from models.Controller.SourceLoaderContext import SourceLoaderContext
from models.Controller.VehicleDetection import VehicleDetection
import cv2

slc: SourceLoaderContext = SourceLoaderContext()
vdet: VehicleDetection = VehicleDetection()
for image in slc.get_source_strategy().get_image_store():
    vdet.annotate_image(image)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
