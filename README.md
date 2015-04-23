------- Various demo/tutorial opencv functions for background subtraction -------

1. I provide the CMakeLists.txt for compilation, tested on ubuntu 64 bit. Should be able to work on windows too (need to compile again on windows). 

2. I also put the binary in the build folder. 

3. Sample run : ./vid_works -h ; Please read comments when "-h" is used. 
   ****NOTE**** - Please provide input (-input FOLDER) and output folder (-output FOLDER) location, at the very least. Output location is needed to save images into that directory. 

4. All 3 samples are in the same main file: mean subtraction, MOG, and MOG by Zoran. Please look at http://docs.opencv.org/modules/video/doc/motion_analysis_and_object_tracking.html for details.

5. Apart from above 3, I also provide a cimg based implementation for removing horizontal lines, along time axis ((temporal_gradient.mp4 in folder part1)). This is a simple basic processing and should not be confused with objective of 4. This approach will only work well when movement is fast (can be parameterized with sigma of deriche filter) and the movement is perpendicular to camera optical axis. Simple line removal is noisy, hence we should try to correlate patches instead of points. 

6. Wherever possible, I try to include the comments so as to understand the code. 


----------------------------------------------------------------------------------------------
Overall impression: I find MOG to be working reasonable from the perspective of detectors. I feel this will help in a good way for object-tracking-based-on-detections (suppression of false positives from detections).
----------------------------------------------------------------------------------------------
	
Questions: ratnesh1729@gmail.com (feel free to shoot)
