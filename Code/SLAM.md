
# RealSense L515 SLAM with ROS Noetic on Ubuntu 20.04  

The realsense L515 camera was deprecated, and is not compatible with new librealsense updates or even realsense-viewer, this is a tutorial on how to do SLAM with the Intel L515 camera in 2025 lol

1. Install Ubuntu 20.04, downgrade kernel to 5.4 

	`sudo apt update`
 `sudo apt install linux-generic`

	Verify it’s there:

	`dpkg --list | grep linux-image`

	You should see linux-image-5.4.0-208-generic (or similar) alongside 5.15.

	`sudo reboot`

Note: When GRUB starts make sure to pick GRUB > “Advanced options” > pick 5.4 
`uname -r` should say 5.4.0-xx.

3.  Install dependencies:
    
    `sudo apt install -y git cmake libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev`
    
4.  Clone and checkout v2.50.0:
    This is the latest realsense lib update that is compatible with L515
    `cd ~/Downloads git clone https://github.com/IntelRealSense/librealsense.git cd librealsense git checkout v2.50.0`
    
5.  Build and install (avoid libusb if on 5.4):
    
    `mkdir build && cd build cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release make -j4 && sudo make install`
    
    -   If USB errors (control_transfer), try `-DFORCE_RSUSB_BACKEND=true` instead, but 5.4 kernel driver worked for us.
6.  Test:
    
    `realsense-viewer`
    
## Step 3: Install ROS Noetic

Ubuntu 20.04 loves Noetic—our SLAM backbone.

1.  Setup repo:

    
    `sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'  sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654`

	` sudo apt update`
    
2.  Install:
    

    
    `sudo apt install -y ros-noetic-desktop-full echo  "source /opt/ros/noetic/setup.bash" >> ~/.bashrc` 
    
    `source ~/.bashrc` 
    `sudo apt install -y python3-rosdep python3-rosinstall` 
    
    `sudo rosdep init rosdep update`
    

## Step 4: Install RealSense ROS Wrapper

Version 2.3.2 matches librealsense v2.50.0.

1.  Create workspace:
  
    `mkdir -p ~/catkin_ws/src`
    `cd ~/catkin_ws/src`
    `git clone -b 2.3.2 https://github.com/IntelRealSense/realsense-ros.git`
    
2.  Install dependency (our ddynamic_reconfigure snag):
   
    
    `sudo apt install -y ros-noetic-ddynamic-reconfigure`
    
3.  Build:
   
    
    `cd ~/catkin_ws` 
    `catkin_make source devel/setup.bash`
    `echo  "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc`
    

## Step 5: Install RTAB-Map

RTAB-Map’s our SLAM engine—needs core libs too.

1.  Install:
    
    `sudo apt install -y ros-noetic-rtabmap ros-noetic-rtabmap-ros`
    ` sudo ldconfig`
    

## Step 6: Run SLAM

Here’s the winning combo—two commands that finally worked after resolution hell.

### Command 1: Launch RealSense Camera

Set both RGB and depth to 640x480 (4:3 aspect ratio) to avoid rgbd_odometry crashes:

`roslaunch realsense2_camera rs_camera.launch \ enable_pointcloud:=true \ enable_color:=true \ enable_depth:=true \ color_width:=640 \ color_height:=480 \ color_fps:=30 \ depth_width:=640 \ depth_height:=480 \ depth_fps:=30`

-   Terminal 1—keep it running.

### Command 2: Launch RTAB-Map SLAM

Hook RTAB-Map to the camera:

`roslaunch rtabmap_ros rtabmap.launch \ rgb_topic:=/camera/color/image_raw \ depth_topic:=/camera/depth/image_rect_raw \ camera_info_topic:=/camera/color/camera_info \ args:="-d" \ rviz:=true \ rtabmapviz:=false`

-   Terminal 2—fires up SLAM and RViz.

## RViz Setup

-   **Fixed Frame**: camera_link
-   **Displays**:
    -   Map > /rtabmap/cloud_map (the SLAM map)
    -   Image > /camera/color/image_raw (RGB feed)
    -   PointCloud2 > /camera/depth/color/points (raw points)
    -   Odometry > /rtabmap/odom (pose)
-   Move the L515—map builds in real-time!
