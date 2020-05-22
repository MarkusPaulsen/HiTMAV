# Class inspired (and code partitially taken) from an implementation by the GitHub user AlfaCodeFlow:
# https://github.com/AlfaCodeFlow/Vehicle_Detection-And-Classification

# <editor-fold desc="Vehicle variables">
colour_for_downwards_vehicles = 0, 0, 255
colour_for_upwards_vehicles = 0, 255, 0
colour_for_undetermined_vehicles = 0, 255, 255
deletion_time = 1.5
# </editor-fold>

# <editor-fold desc="SourceLoaderPolicy variables">
best_strategy = "Video"
# </editor-fold>

# <editor-fold desc="VehicleDetection variables">
top_border = 200
bottom_border = 400
left_border = 100
right_border = 700
colour_recognition_frame = 255, 0, 0
thickness_recognition_frame = 1
colour_bounding_box_centre = 0, 0, 255
thickness_bounding_box_centre = -1
thickness_bounding_box_frame = 2
video_resolution = 800, 450
vehicle_detection_size = 300
# </editor-fold>
