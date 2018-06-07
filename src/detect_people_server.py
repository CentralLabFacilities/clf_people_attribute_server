import rospy
import rospkg
import sys
import os
import cv2
import numpy as np
from threading import Lock

from clf_perception_vision_msgs.srv import LearnPersonImage, DoIKnowThatPersonImage, \
    DoIKnowThatPersonImageRequest, LearnPersonResponse, LearnPersonImageRequest
from gender_and_age_msgs.srv import GenderAndAgeService, GenderAndAgeServiceRequest
from openpose_ros_msgs.msg import PersonAttributesWithPose

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import model_wh, get_graph_path

body_parts = ['Nose',
              'Neck',
              'RightShoulder',
              'RightElbow',
              'RightWrist',
              'LeftShoulder',
              'LeftElbow',
              'LeftWrist',
              'RightHip',
              'RightKnee',
              'RightAnkle',
              'LeftHip',
              'LeftKnee',
              'LeftAnkle',
              'RightEye',
              'LeftEye',
              'RightEar',
              'LeftEar']


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

        cols = int(crop_img.shape[0])
        rows = int(crop_img.shape[1])

        GRID_SIZE = int(np.floor(cols / 10))
        CV_FILLED = -1
        for y in range(0, rows - GRID_SIZE, GRID_SIZE):
            for x in range(0, cols - GRID_SIZE, GRID_SIZE):
                mask = np.full((crop_img.shape[0], crop_img.shape[1]), 0, dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x+GRID_SIZE, y+GRID_SIZE), 255, CV_FILLED)
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
        self.sc = rospy.ServiceProxy(self.topic, GenderAndAgeService)
        self.initialized = True  # todo: wait for service server

    def get_genders_and_ages(self, cropped_images):
        req = GenderAndAgeServiceRequest()
        req.objects = cropped_images
        resp = self.sc.call(req)
        return resp.gender_and_age_response.gender_and_age_list


class FaceID:
    def __init__(self, classify_topic, learn_topic):
        self.learn_topic = learn_topic
        self.classify_topic = classify_topic
        self.learn_face_sc = rospy.ServiceProxy(self.learn_topic, LearnPersonImage)
        self.get_face_name_sc = rospy.ServiceProxy(self.classify_topic, DoIKnowThatPersonImage)
        self.initialized = True  # todo: wait for service server

    def get_name(self, cropped_image):
        req = DoIKnowThatPersonImageRequest()
        req.roi = cropped_image
        r = self.get_face_name_sc.call(req)
        return r.name

    def learn_face(self, cropped_image, name):
        response = LearnPersonResponse()
        req = LearnPersonImageRequest()
        req.roi = cropped_image
        req.name = name
        r = self.learn_face_sc.call(req)
        response.success = r.success
        response.name = r.name
        return response


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def depth_lookup(image, x, y, cx, cy, fx, fy):
        pass

    @staticmethod
    def head_roi(image, person):
        parts = ['Nose', 'RightEar', 'RightEye', 'LeftEar', 'LeftEye']
        amount = int(sum([np.ceil(person[p]['confidence']) for p in parts]))
        if amount <= 1:
            x = y = w = h = -1
            return image[y:y+h, x:x+w]

        vf = sum([np.ceil(person[p]['y']) for p in parts])
        v = int(np.floor(vf/amount))

        # TODO: IMPLEMENT CORRECTLY
        # .... continue with
        # https://github.com/CentralLabFacilities/openpose_ros/blob/pepper_dev/openpose_ros/src/detect_people_server.cpp
        # from line 994

        x = 332
        y = 69
        w = 100
        h = 100

        return image[y:y+h, x:x+w]

    @staticmethod
    def upper_body_roi(image, person):
        parts = ['LeftHip', 'RightHip,', 'LeftShoulder', 'RightShoulder', 'LeftEye']

        # TODO: IMPLEMENT CORRECTLY
        # .... continue with
        # https://github.com/CentralLabFacilities/openpose_ros/blob/pepper_dev/openpose_ros/src/detect_people_server.cpp
        # from line 780

        x = 317
        y = 229
        w = 40
        h = 40
        return image[y:y+h, x:x+w]

    @staticmethod
    def get_posture_and_gesture(person):
        pass


class PoseEstimator:
    def __init__(self, cv_bridge, face_id=None, gender_age=None):

        model = rospy.get_param('~model', 'mobilenet_thin')
        resolution = rospy.get_param('~resolution', '432x368')
        self.resize_out_ratio = float(rospy.get_param('~resize_out_ratio', '4.0'))
        self.face_id = face_id
        self.gender_age = gender_age
        self.cv_bridge = cv_bridge
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

    def get_person_attributes(self, color, depth):

        w = color.shape[0]
        h = color.shape[1]
        acquired = self.tf_lock.acquire(False)
        if not acquired:
            return

        try:
            ts = rospy.Time.now()
            humans = self.humans_to_dict(self.pose_estimator.inference(color, resize_to_default=True,
                                                                       upsample_size=self.resize_out_ratio), w, h)
            rospy.loginfo('timing tf_pose: %r' % (rospy.Time.now()-ts).to_sec())
        finally:
            self.tf_lock.release()

        persons = []
        faces = []
        for human in humans:
            person = PersonAttributesWithPose()
            # Helper.get_posture_and_gesture(human)
            face = self.cv_bridge.cv2_to_imgmsg(Helper.head_roi(color, human), "bgr8")

            ts = rospy.Time.now()
            person.attributes.shirtcolor = ShirtColor.get_shirt_color(Helper.upper_body_roi(color, human))
            rospy.loginfo('timing shirt_color: %r' % (rospy.Time.now() - ts).to_sec())
            faces.append(face)
            if self.face_id is not None and self.face_id.initialized:
                ts = rospy.Time.now()
                person.attributes.name = self.face_id.get_name(face)
                rospy.loginfo('timing face_id: %r' % (rospy.Time.now() - ts).to_sec())
            persons.append(person)
        if self.gender_age is not None and self.gender_age.initialized:
            ts = rospy.Time.now()
            g_a = self.gender_age.get_genders_and_ages(faces)
            rospy.loginfo('timing gender_and_age: %r' % (rospy.Time.now() - ts).to_sec())
            for i in range(0, len(persons)):
                persons[i].attributes.gender_hyp = g_a[i].gender_probability
                persons[i].attributes.age_hyp = g_a[i].age_probability

        rospy.loginfo(persons)
        return persons

    def get_closest_person_face(self, color, depth):
        persons = self.get_person_attributes(color, depth)
        # TODO: filter closest
        return Helper.head_roi(color, persons[0])

    @staticmethod
    def humans_to_dict(humans, w, h):
        persons = []
        for human in humans:
            person = {}
            for part in body_parts:
                try:
                    body_part = human.body_parts[body_parts.index(part)]
                    person[part] = {'confidence': body_part.score,
                                    'x': body_part.x * w,
                                    'y': body_part.y * h}
                except KeyError:
                    person[part] = {'confidence': 0.0,
                                    'x': 0,
                                    'y': 0}
                # print(person[part])
            persons.append(person)
        return persons
