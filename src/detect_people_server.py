import rospy
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
    def get_shirt_color(cropped_image):
        pass

    @staticmethod
    def get_pixel_color_type(pixel):
        pass


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
        pass

    def get_persons(self, color, depth):
        pass

    def get_closest_person_face(self, color, depth):
        pass
