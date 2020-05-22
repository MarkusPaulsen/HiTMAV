# HiTMAV
Highway Traffic Monitoring using Aerial Video

To start the system:
1. Place a video or an image series into the data folder (you can remove the placeholder).
2. Adjust the parameters in Configuration.py. Adjust especially:
    - Variable best_strategy: If you want to read an image series, set it to Image, else, set it to video.
    - Variable top_border, bottom_border, left_border, right_border: Set the box borders in a way, that the area, in which you want to detect vehicles is inside the box.
3. Start the script Client.py