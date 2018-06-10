#!/usr/bin/env python
import rospy
from detect_people_server import FaceID, GenderAndAge, PoseEstimator
from pepper_clf_msgs.srv import DepthAndColorImage
from openpose_ros_msgs.srv import GetCrowdAttributesWithPose, GetCrowdAttributesWithPoseResponse, GetFollowRoi, \
    GetFollowRoiResponse
from clf_perception_vision_msgs.srv import LearnPerson, LearnPersonResponse
from cv_bridge import CvBridge, CvBridgeError


class PeopleAttributeServer:
    def __init__(self):
        rospy.loginfo('>>init PeopleAttributeServer')
        self.image_topic = '/naoqi_driver/get_images'
        self.crowd_topic = '/open_pose/get_crowd_attributes'
        self.learn_topic = '/open_pose/learn_face'
        self.follow_topic = '/open_pose/shirt_roi'

        self.face_know_topic = 'clf_face_identification_know_image'
        self.face_learn_topic = 'clf_face_identification_learn_image'
        self.gender_age_topic = 'clf_gender_age_classify_array'
        rospy.init_node('people_attribute_server')

        self.image_grabber = rospy.ServiceProxy(self.image_topic, DepthAndColorImage)
        self.crowd_service = rospy.Service(self.crowd_topic, GetCrowdAttributesWithPose, self.detect_crowd)
        self.learn_service = rospy.Service(self.learn_topic, LearnPerson, self.learn_face)
        self.follow_roi_service = rospy.Service(self.follow_topic, GetFollowRoi, self.get_follow_roi)

        self.face_id = FaceID(self.face_know_topic, self.face_learn_topic)
        self.gender_age = GenderAndAge(self.gender_age_topic)

        self.cv_bridge = CvBridge()
        self.estimator = PoseEstimator(cv_bridge=self.cv_bridge, face_id=self.face_id, gender_age=self.gender_age)
        rospy.loginfo('pose_estimator ready')

    def detect_crowd(self, request):
        response = GetCrowdAttributesWithPoseResponse()
        image = self.image_grabber.call()
        try:
            color = self.cv_bridge.imgmsg_to_cv2(image.color, "bgr8")
            depth = self.cv_bridge.imgmsg_to_cv2(image.depth, "32FC1")
            persons = self.estimator.get_person_attributes(color, depth)
            response.attributes = persons
            return response
        except CvBridgeError as e:
            rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
        return

    def learn_face(self, request):
        image = self.image_grabber.call()
        try:
            color = self.cv_bridge.imgmsg_to_cv2(image.color, "bgr8")
            depth = self.cv_bridge.imgmsg_to_cv2(image.depth, "32FC1")
            face = self.cv_bridge.cv2_to_imgmsg(self.estimator.get_closest_person_face(color, depth))
            response = self.face_id.learn_face(face, request.name)
            return response
        except CvBridgeError as e:
            rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
            return LearnPersonResponse()

    def get_follow_roi(self, request):
        image = self.image_grabber.call()
        try:
            color = self.cv_bridge.imgmsg_to_cv2(image.color, "bgr8")
            depth = self.cv_bridge.imgmsg_to_cv2(image.depth, "32FC1")
            response = GetFollowRoiResponse()
            response.roi = self.estimator.get_closest_person_body_roi(color, depth)
            return response
        except CvBridgeError as e:
            rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
            return GetFollowRoiResponse()


if __name__ == "__main__":
    server = PeopleAttributeServer()
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException, ex:
            rospy.logwarn(">>> Exiting ... %s" % str(ex))
