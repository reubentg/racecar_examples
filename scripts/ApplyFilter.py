#!/usr/bin/env python

import rospy
import numpy as np
from scipy import signal
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Applies a filter to images received on a specified topic and publishes the filtered image
class Filter:
    # Initialize the filter
    def __init__(self, filter_path, sub_topic, pub_topic):
        # Load the filter from csv file
        self.filter = np.loadtxt(open(filter_path, 'rb'), delimiter=',')
        if len(self.filter.shape) < 2: # If filter is one-dimensional, add axis
            self.filter = self.filter[np.newaxis,:]
        self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb, queue_size=5)
        self.pub_red = rospy.Publisher(pub_topic_red, Image, queue_size=1)
        self.pub_blue = rospy.Publisher(pub_topic_blue, Image, queue_size=1)
        self.bridge = CvBridge()

    def apply_filter_cb(self, msg):
        in_image = None
        try:
            in_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv2.cvtColor(in_image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0,50, 50])
        upper_red= np.array([14, 255, 255])
        mask_red0 = cv2.inRange(img, lower_red, upper_red)

        lower_red = np.array([160,50, 50])
        upper_red= np.array([180, 255, 255])
        mask_red1 = cv2.inRange(img, lower_red, upper_red)

        mask_red = (mask_red0 + mask_red1)

        # define range of blue color in HSV
        lower_blue = np.array([95, 105, 20])
        upper_blue = np.array([115, 255, 255])

        # mask for the colors
        mask_blue = cv2.inRange(img, lower_blue, upper_blue)

        thresh_blue = cv2.threshold(mask_blue, 127, 255, cv2.THRESH_BINARY)[1]
        #    thresh = cv2.threshold(mask_blue,60,255,cv2.THRESH_BINARY)[1]

        cnts_blue = cv2.findContours(thresh_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        thresh_red = cv2.threshold(mask_red, 127, 255, cv2.THRESH_BINARY)[1]
        #    thresh = cv2.threshold(mask_blue,60,255,cv2.THRESH_BINARY)[1]

        cnts_red = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(cnts_blue) > 5:
            maxcontour = max(cnts_blue, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(maxcontour)
            center = (int(x), int(y))
            a = np.arctan2(x - 320, 480 - y)

            print 'Blue angle:%f Blue radius:%f ' % (a, radius)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame, then update the list of tracked points
                cv2.circle(in_image, center, int(radius), (0, 255, 255), 2)
                cv2.circle(in_image, center, 5, (0, 0, 255), -1)

        for c in cnts_blue:
            cv2.drawContours(in_image, [c], -1, (0, 255, 0), 1)  # draws the Conture lines
            cv2.drawContours(thresh_blue, [c], -1, (0, 255, 0), 1)

        try:
            self.pub_blue.publish(self.bridge.cv2_to_imgmsg(in_image, encoding="passthrough"))
        except CvBridgeError as e:
            print e

        if len(cnts_red) > 5:
            maxcontour = max(cnts_red, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(maxcontour)
            center = (int(x), int(y))
            a = np.arctan2(x - 320, 480 - y)

            print 'Red angle:%f Red radius:%f ' % (a, radius)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame, then update the list of tracked points
                cv2.circle(in_image, center, int(radius), (0, 0, 255), 2)
                cv2.circle(in_image, center, 5, (0, 255, 0), -1)

        for c in cnts_red:
            cv2.drawContours(in_image, [c], -1, (0, 255, 0), 1)  # draws the Conture lines
            cv2.drawContours(thresh_red, [c], -1, (0, 255, 0), 1)

        # Display the resulting frame
        try:
            self.pub_red.publish(self.bridge.cv2_to_imgmsg(in_image, encoding="passthrough"))
            # self.pub_blue.publish(self.bridge.cv2_to_imgmsg(in_image, encoding="passthrough"))
        except CvBridgeError as e:
            print e

    def apply_filter_cb_old(self, msg):
        try:
            in_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        img = cv2.cvtColor(in_image, cv2.COLOR_BGR2HSV)
        # height, width, channels = img.shape #480 x 640
        # print height, width, channels

        # define range of red color in HSV NOT THE BEST ON THE FLOOR
        lower_red = np.array([0,50, 50])
        upper_red= np.array([14, 255, 255])
        mask_red0 = cv2.inRange(img, lower_red, upper_red)

        lower_red = np.array([160,50, 50])
        upper_red= np.array([180, 255, 255])
        mask_red1 = cv2.inRange(img, lower_red, upper_red)

        # define range of blue color in HSV
        lower_blue = np.array([95,105, 20])
        upper_blue = np.array([115, 255, 255])

        #mask for the colors
        mask_blue = cv2.inRange(img, lower_blue, upper_blue)
        mask_red = (mask_red0 + mask_red1)

        mask_red_blur = cv2.GaussianBlur(mask_red, (11,11), 0 )
        mask_blue_blur = cv2.GaussianBlur(mask_blue, (11, 11), 0)

        status = cv2.imwrite('/home/nvidia/blue.png', self.bridge.cv2_to_compressed_imgmsg(mask_blue_blur, 'png'))
        print status

        cropped_blue = np.zeros[6]

        cropped_blue[0] = np.count_nonzero(mask_blue_blur[0:480, 0:150])
        cropped_blue[1] = np.count_nonzero(mask_blue_blur[0:480, 151:250])
        cropped_blue[2] = np.count_nonzero(mask_blue_blur[0:480, 251:300])

        cropped_blue[3] = np.count_nonzero(mask_blue_blur[0:480, 300:340])

        cropped_blue[4] = np.count_nonzero(mask_blue_blur[0:480, 341:390])
        cropped_blue[5] = np.count_nonzero(mask_blue_blur[0:480, 391:490])
        cropped_blue[6] = np.count_nonzero(mask_blue_blur[0:480, 491:640])

        cropped_red = mask_red_blur[0:300, 300:340]

        cropped_blue = np.count_nonzero(cropped_blue)

        print 'cropped_blue: '
        print str(cropped_blue).strip('[]')
        print 'red: '
        print str(cropped_red)

        cropped_fl_blue = mask_blue_blur[0:480, 0:150]
        cropped_l_blue = mask_blue_blur[0:480, 151:300]
        cropped_c_blue = mask_blue_blur[0:480, 300:340]
        cropped_r_blue = mask_blue_blur[0:480, 340:490]
        cropped_fr_blue = mask_blue_blur[0:480, 491:640]

        cropped_blue= cv2.countNonZero(cropped_blue)



        try:
            self.pub_red.publish(self.bridge.cv2_to_imgmsg(mask_red_blur, encoding="passthrough"))
            self.pub_blue.publish(self.bridge.cv2_to_imgmsg(mask_blue_blur, encoding="passthrough"))
        except CvBridgeError as e:
            print e
    
    
    # Callback for when an image is received. Applies the filter to that image
    def apply_filter_cb_pat(self, msg):
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
        except CvBridgeError as e:
            print(e)

        
if __name__ == '__main__':
    filter_path = None # The path to a csv file containing the filter to be applied
    sub_topic = None # The image topic to apply the filter to
    pub_topic = None # The topic to publish filtered images to
    
    rospy.init_node('apply_filter', anonymous=True)
    
    # Populate params with values passed by launch file
    filter_path = rospy.get_param("~filter_path", None)
    sub_topic = rospy.get_param("~sub_topic", None)
    pub_topic_red = rospy.get_param("~pub_topic_red", None)
    pub_topic_blue = rospy.get_param("~pub_topic_blue", None)
    
    f = Filter(filter_path, sub_topic, pub_topic)
    rospy.spin()
