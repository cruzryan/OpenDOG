
# RealSense L515 SLAM with ROS Noetic on Ubuntu 20.04  
1. This makes L515 compatible
2.  Install dependencies:
    
    bash
    
    CollapseWrapCopy
    
    `sudo apt install -y git cmake libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev`
    
3.  Clone and checkout v2.50.0:
    
    bash
    
    CollapseWrapCopy
    
    `cd ~/Downloads git clone https://github.com/IntelRealSense/librealsense.git cd librealsense git checkout v2.50.0`
    
4.  Build and install (avoid libusb if on 5.4):
    
    bash
    
    CollapseWrapCopy
    
    `mkdir build && cd build cmake .. -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release make -j4 && sudo make install`
    
    -   If USB errors (control_transfer), try -DFORCE_RSUSB_BACKEND=true instead, but 5.4 kernel driver worked for us.
5.  Test:
    
    bash
    
    CollapseWrapCopy
    
    `realsense-viewer`
    
    -   Plug in L515—RGB and depth should stream. Save 640x480 config if needed (gear > Save).

## Step 3: Install ROS Noetic

Ubuntu 20.04 loves Noetic—our SLAM backbone.

1.  Setup repo:
    
    bash
    
    CollapseWrapCopy
    
    `sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'  sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 sudo apt update`
    
2.  Install:
    
    bash
    
    CollapseWrapCopy
    
    `sudo apt install -y ros-noetic-desktop-full echo  "source /opt/ros/noetic/setup.bash" >> ~/.bashrc source ~/.bashrc sudo apt install -y python3-rosdep python3-rosinstall sudo rosdep init rosdep update`
    

## Step 4: Install RealSense ROS Wrapper

Version 2.3.2 matches librealsense v2.50.0.

1.  Create workspace:
    
    bash
    
    CollapseWrapCopy
    
    `mkdir -p ~/catkin_ws/src cd ~/catkin_ws/src git clone -b 2.3.2 https://github.com/IntelRealSense/realsense-ros.git`
    
2.  Install dependency (our ddynamic_reconfigure snag):
    
    bash
    
    CollapseWrapCopy
    
    `sudo apt install -y ros-noetic-ddynamic-reconfigure`
    
3.  Build:
    
    bash
    
    CollapseWrapCopy
    
    `cd ~/catkin_ws catkin_make source devel/setup.bash echo  "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc`
    

## Step 5: Install RTAB-Map

RTAB-Map’s our SLAM engine—needs core libs too.

1.  Install:
    
    bash
    
    CollapseWrapCopy
    
    `sudo apt install -y ros-noetic-rtabmap ros-noetic-rtabmap-ros sudo ldconfig`
    

## Step 6: Run SLAM

Here’s the winning combo—two commands that finally worked after resolution hell.

### Command 1: Launch RealSense Camera

Set both RGB and depth to 640x480 (4:3 aspect ratio) to avoid rgbd_odometry crashes:

bash

CollapseWrapCopy

`roslaunch realsense2_camera rs_camera.launch \ enable_pointcloud:=true \ enable_color:=true \ enable_depth:=true \ color_width:=640 \ color_height:=480 \ color_fps:=30 \ depth_width:=640 \ depth_height:=480 \ depth_fps:=30`

-   Terminal 1—keep it running.

### Command 2: Launch RTAB-Map SLAM

Hook RTAB-Map to the camera:

bash

CollapseWrapCopy

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
