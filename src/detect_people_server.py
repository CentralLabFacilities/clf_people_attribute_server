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
        vf = sum([np.ceil(person[p]['y']) for p in parts])
        v = int(np.floor(vf/amount))

        if amount <= 1:
            x = y = w = h = -1
            return image[y:y+h, x:x+w]

        distlist_x = []
        if person['Nose']['x'] != 0:
            distlist_x.append(np.abs(person['Nose']['x']))
        if person['RightEar']['x'] != 0:
            distlist_x.append(np.abs(person['RightEar']['x']))
        if person['RightEye']['x'] != 0:
            distlist_x.append(np.abs(person['RightEye']['x']))
        if person['LeftEar']['x'] != 0:
            distlist_x.append(np.abs(person['LeftEar']['x']))
        if person['LeftEye']['x'] != 0:
            distlist_x.append(np.abs(person['LeftEye']['x']))

        max_dist_u = np.amax(distlist_x)

        min_dist_u = np.amin(distlist_x)

        x = min_dist_u
        w = max_dist_u - min_dist_u
        h = w * 1.5
        y = v - h / 2

        if (x + w >= image.shape[0]):
            w = w - np.abs((x + w) - image.shape[0])
        if (y + h >= image.shape[1]):
            h = h - np.abs((y + h) - image.shape[1])

        return image[int(y):int(y+h), int(x):int(x+w)]

    @staticmethod
    def upper_body_roi(image, person):
        parts = ['LeftShoulder', 'RightShoulder', 'LeftHip', 'RightHip']
        amount = int(sum([np.ceil(person[p]['confidence']) for p in parts]))

        if amount <= 1:
            x = y = w = h = -1
            return image[y:y+h, x:x+w]

        if person['LeftShoulder']['confidence'] > 0 and person['RightShoulder']['confidence'] > 0:
            y = person['RightShoulder']['y']
            w = np.abs(person['LeftShoulder']['y'] - person['RightShoulder']['x'])
            if (person['RightShoulder']['x'] - person['LeftShoulder']['x']) < 0:
                x = person['RightShoulder']['x']
            else:
                x = person['LeftShoulder']['x']

            if person['RightHip']['confidence'] > 0:
                h = (person['RightHip']['y'] - y)
            elif (person['LeftHip']['confidence'] > 0):
                h = (person['LeftHip']['y'] - y)
            else:
                rospy.logerr("no hip found")
                h = w
        else:
            if (person['RightHip']['confidence'] > 0 and person['LeftHip']['confidence'] > 0):
                if ((person['RightShoulder']['confidence'] > 0) ^ (person['LeftShoulder']['confidence'] > 0)):
                    w = np.abs(person['LeftHip']['x'] - person['RightHip']['x'])
                    y = person['LeftShoulder']['y'] + person['RightShoulder']['y']
                    if ((person['RightHip']['x'] - person['LeftHip']['x']) < 0):
                        x = person['RightHip']['x']
                        h = person['RightHip']['y'] - person['LeftShoulder']['x'] - person['RightShoulder']['x'] #one of the shoulders values will be 0.
                    else:
                        x = person['LeftHip']['x']
                        h = person['LeftHip']['y'] - person['LeftShoulder']['x'] - person['RightShoulder']['x']
                else:
                    w = np.abs(person['LeftHip']['x'] - person['RightHip']['x'])
                    if ((person['RightHip']['x'] - person['LeftHip']['x']) < 0):
                        x = person['RightHip']['x']
                        h = w
                        y = person['RightHip']['y'] - h
                    else:
                        x = person['LeftHip']['x']
                        h = w
                        y = person['LeftHip']['y'] - h
            else:
                if (person['RightHip']['confidence'] > 0):
                    if (person['RightShoulder']['confidence'] > 0):
                        x = person['RightShoulder']['x']
                        y = person['RightShoulder']['y']
                        h = np.abs(person['RightShoulder']['y'] - person['RightHip']['y'])
                        w = h * 0.5
                    
                    if (person['LeftShoulder']['confidence'] > 0):
                        x = person['RightHip']['x']
                        y = person['LeftShoulder']['y']
                        h = np.abs(person['LeftShoulder']['y'] - person['RightHip']['y'])
                        w = np.abs(person['LeftShoulder']['x'] - person['RightHip']['x'])
                if (person['LeftHip']['confidence'] > 0):
                    if (person['RightShoulder']['confidence'] > 0):
                        x = person['RightShoulder']['x']
                        y = person['RightShoulder']['y']
                        h = np.abs(person['RightShoulder']['y'] - person['LeftHip']['y'])
                        w = h * 0.5
                    
                    if (person['LeftShoulder']['confidence'] > 0):
                        y = person['LeftShoulder']['y']
                        h = np.abs(person['LeftShoulder']['y'] - person['LeftHip']['y'])
                        w = h * 0.5
                        x = person['LeftShoulder']['x'] - w
                else:
                    rospy.log("No BB possible: RShoulder['confidence'] %f, LShoulder['confidence'] %f,"
                              " RHip['confidence'] %f, LHip['confidence'] %f \n",
                              person['RightShoulder']['confidence'], person['LeftShoulder']['confidence'], person['RightHip']['confidence'],
                              person['LeftHip']['confidence'])

                
        
            

        if (x + w >= image.shape[0]):
            w = w - np.abs((x + w) - image.shape[0])
        if (y + h >= image.shape[1]):
            h = h - np.abs((y + h) - image.shape[1])

        if ((w <= 0) or (h <= 0) or (x <= 0) or (y <= 0)):
            rospy.log("w or h <= 0")
            x = y = w = h = 0

        return image[int(y):int(y+h), int(x):int(x+w)]

    @staticmethod
    def get_posture_and_gestures(person):
        posture = 2  # 2 = standing
        gestures = [6]  # 6 = neutral
        return {'posture': posture, 'gestures': gestures}


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

            pg = Helper.get_posture_and_gestures(human)
            person.attributes.posture = pg['posture']
            person.attributes.gestures = pg['gestures']

            face = self.cv_bridge.cv2_to_imgmsg(Helper.head_roi(color, human), "bgr8")
            faces.append(face)

            ts = rospy.Time.now()
            person.attributes.shirtcolor = ShirtColor.get_shirt_color(Helper.upper_body_roi(color, human))
            rospy.loginfo('timing shirt_color: %r' % (rospy.Time.now() - ts).to_sec())

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
