# Machine-Perception
A collection of my Computer Vision Projects from CIS 5800

Project 1: Logo Projection with Homographies in Sports Footage
Overview:
In this project, I utilized projective geometry and homography concepts to project the Penn Engineering logo onto the goalposts in a sequence of football match images. This allowed the logo to align with the goal dynamically, preserving perspective consistency across video frames.

Key Components:

Homography Estimation: Estimated a homography matrix to map the Penn logo’s corners onto the specified corners of the goalpost in each frame.
Inverse Warping for Logo Projection: By warping the points from each video frame to the logo space, inverse warping ensured that every pixel in the frame was correctly mapped to a pixel in the logo, preserving a seamless overlay without "holes."
Automated Projection: Automated the process for each frame in the sequence, creating a continuous, perspective-accurate projection across the video.

Run the project by executing project_logo.py to view the logo projected onto the football goal throughout the video.

Project 2: Augmented Reality with AprilTags

Overview:
This project implemented a simple augmented reality application using AprilTags to determine the camera’s position and orientation in each video frame. Virtual objects could then be placed accurately within the scene based on the computed camera pose.

Key Components:

World Coordinate Setup: Established a world coordinate system centered on the AprilTag to serve as a reference frame for the camera.
Pose Estimation:
- PnP Method: Used homography-based Perspective-n-Point (PnP) estimation for initial camera pose recovery.
- P3P and Procrustes Analysis: Refined pose estimation by calculating camera pose using the Perspective-Three-Point (P3P) approach, with Procrustes analysis to solve for accurate rotation and translation.
3D Virtual Object Placement: With precise 3D camera pose established, arbitrary objects could be augmented into the video, appearing fixed within the scene.

To run, execute main.py to see virtual objects appear anchored within the video scene, utilizing real-world coordinates for natural placements.
