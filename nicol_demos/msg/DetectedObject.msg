string name                   # Name of the object (e.g. tomato soup can)
uint64 track_id               # Tracking ID of the detected object (should survive blips of false or non-detection)
geometry_msgs/Point32 centre  # 3D world coordinate of the geometric centre of the object
geometry_msgs/Point32 bottom  # 3D world coordinate of the bottom of the object projected to table height
float32 pitch                 # Target head pitch angle in radians that would make the camera look directly at the object
float32 yaw                   # Target head yaw angle in radians that would make the camera look directly at the object
Pair centre_px                # 2D pixel coordinates of the centre of the object based on the detected bounding box
Pair bottom_px                # 2D pixel coordinates of the bottom of the object based on the detected bounding box
Pair frame_resolution         # width and height of the node's input frame (in pixels)
Pair frame_size
