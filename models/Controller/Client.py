import cv2

from models.Controller.Configuration import *
from models.Model.SourceLoaderContext import SourceLoaderContext
from models.View.ImageOutput import ImageOutput
from models.Controller.VehicleManager import VehicleManager
from models.Controller.BoundingBoxCreator import BoundingBoxCreator

slc: SourceLoaderContext = SourceLoaderContext()
io: ImageOutput = ImageOutput()
vm: VehicleManager = VehicleManager()
bbc: BoundingBoxCreator = BoundingBoxCreator()
sorted_image_store = slc.get_source_strategy().get_image_store()
if best_strategy == "Image":
    sorted_image_store.sort(key=(lambda image: image.get_image_name()))
for image in sorted_image_store:
    print(image.get_image_name())
    bbc.compute_next_image(image.get_image_data())
    vm.update_vehicles(bbc.get_bounding_box_centre_points(), bbc.get_bounding_box_frame_points())
    io.set_annotated_resized_image(bbc.get_resized_image())
    io.set_annotated_background_subtract(bbc.get_background_subtract())
    io.set_annotated_close_transformation(bbc.get_close_transformation())
    io.set_vehicle_list(vm.get_vehicle_list())
    io.annotate()
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
