#!/usr/bin/env python
import rospy
from detect_people_server import FaceID, GenderAndAge, PoseEstimator


from pepper_clf_msgs.srv import DepthAndColorImage
from openpose_ros_msgs.srv import GetCrowdAttributes, GetCrowdAttributesResponse
from clf_perception_vision_msgs.srv import LearnPerson, LearnPersonResponse


class PeopleAttributeServer:
    def __init__(self):
        rospy.loginfo('>>init PeopleAttributeServer')
        self.image_topic = '/naoqi_driver/get_images'
        self.crowd_topic = '/open_pose/get_crowd_attributes'
        self.learn_topic = '/open_pose/learn_face'

        self.face_know_topic = 'clf_face_identification_know_image'
        self.face_learn_topic = 'clf_face_identification_learn_image'
        self.gender_age_topic = 'clf_gender_age_classify_array'
        rospy.init_node('people_attribute_server', anonymous=True)

        self.image_grabber = rospy.ServiceProxy(self.image_topic, DepthAndColorImage)
        self.crowd_service = rospy.Service(self.crowd_topic, GetCrowdAttributes, self.detect_crowd)
        self.learn_service = rospy.Service(self.learn_topic, LearnPerson, self.learn_face)

        self.face_id = FaceID(self.face_know_topic, self.face_learn_topic)
        self.gender_age = GenderAndAge(self.gender_age_topic)
        self.estimator = PoseEstimator()

    def detect_crowd(self):
        response = GetCrowdAttributesResponse()
        image = self.image_grabber.call()
        response.attributes = self.estimator.get_persons(image.color, image.depth)
        return response

    def learn_face(self, request):
        response = LearnPersonResponse()
        image = self.image_grabber.call()
        face = self.estimator.get_closest_person_face(image.color, image.depth)
        response.success = self.face_id.learn_face(face, request.name)
        return response


if __name__ == "__main__":

    server = PeopleAttributeServer()
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException, ex:
            rospy.logwarn(">>> Exiting ... %s" % str(ex))
