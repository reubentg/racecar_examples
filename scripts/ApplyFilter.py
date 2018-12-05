#!/usr/bin/env python

import rospy
import numpy as np
from scipy import signal
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Applies a filter to images received on a specified topic and publishes the filtered image
class Filter:
    # Initialize the filter
    def __init__(self, filter_path, sub_topic, pub_topic):
        # Load the filter from csv file
        self.filter = np.loadtxt(open(filter_path, 'rb'), delimiter=',')
        if len(self.filter.shape) < 2: # If filter is one-dimensional, add axis
            self.filter = self.filter[np.newaxis,:]
        self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb, queue_size=5)
        self.pub = rospy.Publisher(pub_topic, Image, queue_size=1)
        self.bridge = CvBridge()
    
    # Callback for when an image is received. Applies the filter to that image
    def apply_filter_cb(self, msg):
        # Convert the image to a numpy array
        try:
            in_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
            
        in_shape = in_image.shape
        out_shape = (in_shape[0]-self.filter.shape[0]+1, 
                     in_shape[1]-self.filter.shape[1]+1, 
                     in_shape[2])
        out_image = np.ndarray(shape=out_shape, dtype=in_image.dtype)

        # Apply filter to each channel of the image
        for ch in xrange(out_shape[2]):
            out_image[:,:,ch] = signal.convolve2d(in_image[:,:,ch],self.filter,'valid')
        
        # Publish the resulting image
        try:
            self.pub.publish(self.bridge.cv2_to_imgmsg(out_image, encoding="passthrough"))
        except CvBridgeErroor as e:
            print(e)
        
if __name__ == '__main__':
    filter_path = None # The path to a csv file containing the filter to be applied
    sub_topic = None # The image topic to apply the filter to
    pub_topic = None # The topic to publish filtered images to
    
    rospy.init_node('apply_filter', anonymous=True)
    
    # Populate params with values passed by launch file
    filter_path = rospy.get_param("~filter_path", None)
    sub_topic = rospy.get_param("~sub_topic", None)
    pub_topic = rospy.get_param("~pub_topic", None)
    
    f = Filter(filter_path, sub_topic, pub_topic)
    rospy.spin()
