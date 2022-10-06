# Import packages
import cv2
import time
from threading import Thread
import os
from datetime import datetime

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self


    """def write(self):
        today = datetime.now()
        today_str = today.strftime("%Y%m%d")
        time_str = today.strftime("%H%m%s")
        video_path = r'Coater_Video/video'
        video_path_today = os.path.join(video_path, today_str)

        if not os.path.exists(video_path_today):
            os.mkdir(video_path_today)

        videofile = 'output'+today_str+'-'+time_str+'.avi'
        videofile_path = os.path.join(video_path_today, videofile)
        print(videofile_path)

        self.out = cv2.VideoWriter(videofile_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (640,480))
        Thread(target=self.update,args=()).start()
        return self """

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

#    def writeout(self):
#        self.out.write(self.frame)
    
    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
