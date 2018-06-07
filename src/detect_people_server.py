import rospy
import rospkg
import sys
import os
import cv2
import numpy as np
from threading import Lock
from clf_perception_vision_msgs.srv import LearnPersonImage, DoIKnowThatPersonImage
from gender_and_age_msgs.srv import GenderAndAgeService

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from tfpose_ros.msg import Persons, Person, BodyPartElm

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import model_wh, get_graph_path


class ShirtColor:
    def __init__(self):
        pass

    @staticmethod
    def get_shirt_color(crop_img):

        hsv_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        bin_colors = dict()

        bin_colors["white"] = 0
        bin_colors["black"] = 0
        bin_colors["grey"] = 0
        bin_colors["red"] = 0
        bin_colors["orange"] = 0
        bin_colors["yellow"] = 0
        bin_colors["green"] = 0
        bin_colors["cyan"] = 0
        bin_colors["blue"] = 0
        bin_colors["purple"] = 0

        GRID_SIZE = np.floor(crop_img.cols / 10)

        for y in range(0, crop_img.rows - GRID_SIZE, GRID_SIZE):
            for x in range(0, crop_img.cols - GRID_SIZE, GRID_SIZE):
                mask = cv2.zeros(crop_img.size, cv2.CV_8UC1)
                bin_colors[ShirtColor.get_pixel_color_type(cv2.mean(hsv_crop_img, mask))] += 1

        result_color = "no color"
        max_bin_count = 0

        for key, value in bin_colors.iteritems():
            if max_bin_count < value:
                result_color = key
                max_bin_count = value

        return result_color

    @staticmethod
    def get_pixel_color_type(hsv_val):

        H = hsv_val[0]
        S = hsv_val[1]
        V = hsv_val[2]

        color = "no color"

        if V < 75:
            color = "black"
        elif V > 190 and S < 27:
            color = "white"
        elif S < 53 and V < 185:
            color = "grey"
        elif H < 14:
            color = "red"
        elif H < 25:
            color = "orange"
        elif H < 34:
            color = "yellow"
        elif H < 73:
            color = "green"
        elif H < 102:
            color = "cyan"
        elif H < 127:
            color = "blue"
        elif H < 149:
            color = "purple"
        else:
            color = "red"

        return color


class GenderAndAge:
    def __init__(self, topic):
        self.topic = topic

    def get_gender_and_age(self, cropped_image):
        pass


class FaceID:
    def __init__(self, learn_topic, classify_topic):
        self.learn_topic = learn_topic
        self.classify_topic = classify_topic
        self.learn_face_sc = rospy.ServiceProxy(self.learn_topic, LearnPersonImage)
        self.get_face_name_sc = rospy.ServiceProxy(self.classify_topic, DoIKnowThatPersonImage)

    def get_name(self, cropped_image):
        pass

    def learn_face(self, cropped_image, name):
        pass


class Helper:
    def __init__(self):
        pass

    def depth_lookup(self, image, x, y, cx, cy, fx, fy):
        pass

    def head_roi(self, image, person):
        pass

    def upper_body_roi(self, image, person):
        pass

    def get_posture_and_gesture(self, person):
        pass


class PoseEstimator:
    def __init__(self):

        model = rospy.get_param('~model', 'cmu')
        resolution = rospy.get_param('~resolution', '432x368')
        self.resize_out_ratio = float(rospy.get_param('~resize_out_ratio', '4.0'))
        self.tf_lock = Lock()

        try:
            w, h = model_wh(resolution)
            graph_path = get_graph_path(model)
            rospack = rospkg.RosPack()
            graph_path = os.path.join(rospack.get_path('tfpose_ros'), graph_path)
        except Exception as e:
            rospy.logerr('invalid model: %s, e=%s' % (model, e))
            sys.exit(-1)

        self.pose_estimator = TfPoseEstimator(graph_path, target_size=(w, h))

    @staticmethod
    def get_humans(self, color):

        acquired = self.tf_lock.acquire(False)
        if not acquired:
            return

        try:
            humans = self.pose_estimator.inference(color, resize_to_default=True, upsample_size=self.resize_out_ratio)
        finally:
            self.tf_lock.release()

        return humans

    def get_person_attributes(self, color, depth):

        humans = self.get_humans(color)

        pass

    def get_closest_person_face(self, color, depth):

        humans = self.get_humans(color)

        pass
