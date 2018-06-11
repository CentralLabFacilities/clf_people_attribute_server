import rospy
import rospkg
import sys
import os
import cv2
import numpy as np

from enum import Enum
from threading import Lock, Thread

from clf_perception_vision_msgs.srv import LearnPersonImage, DoIKnowThatPersonImage, \
    DoIKnowThatPersonImageRequest, LearnPersonResponse, LearnPersonImageRequest
from gender_and_age_msgs.srv import GenderAndAgeService, GenderAndAgeServiceRequest
from openpose_ros_msgs.msg import PersonAttributesWithPose
from sensor_msgs.msg import CameraInfo, Image, RegionOfInterest
from geometry_msgs.msg import PoseStamped

from tf import TransformListener

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


class Gesture(Enum):
    POINTING_LEFT = 1
    POINTING_RIGHT = 2
    RAISING_LEFT_ARM = 3
    RAISING_RIGHT_ARM = 4
    WAVING = 5
    NEUTRAL = 6

class Posture(Enum):
    SITTING = 1
    STANDING = 2
    LYING = 3

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
                cv2.rectangle(mask, (x, y), (x + GRID_SIZE, y + GRID_SIZE), 255, CV_FILLED)
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

        def init_service():
            try:
                rospy.loginfo('wait for gender_age...')
                rospy.wait_for_service(self.topic, 3.0)
                self.initialized = True
            except rospy.ROSException as e:
                rospy.loginfo(e)
                self.initialized = False
            rospy.loginfo('<<< GenderAndAge: %r' % self.initialized)

        t = Thread(target=init_service())
        t.start()

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

        def init_service():
            try:
                rospy.loginfo('wait for face_id...')
                rospy.wait_for_service(self.classify_topic, 3.0)
                self.initialized = True
            except rospy.ROSException as e:
                rospy.loginfo(e)
                self.initialized = False
            rospy.loginfo('<<< FaceID: %r' % self.initialized)

        t = Thread(target=init_service())
        t.start()

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
    def __init__(self, depth_info_topic='/pepper_robot/camera/depth/camera_info'):

        self.tf = TransformListener()
        self.depth_sub = rospy.Subscriber(depth_info_topic,
                                          CameraInfo, self.camera_info_callback)
        self.camera_frame = 'CameraDepth_optical_frame'
        self.base_frame = 'base_footprint'
        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.cx = msg.K[2]
        self.fy = msg.K[4]
        self.cy = msg.K[5]
        rospy.loginfo('camera info: %r' % msg)
        self.depth_sub.unregister()

    def depth_lookup(self, color_image, depth_image, crx, cry, crw, crh, time_stamp, is_in_mm):
        w_factor = (float(depth_image.shape[1]) / float(color_image.shape[1]))
        h_factor = (float(depth_image.shape[0]) / float(color_image.shape[0]))
        x = crx * w_factor
        y = cry * h_factor
        w = crw * w_factor
        h = crh * h_factor

        rospy.loginfo('is in mm: %r' % is_in_mm)
        unit_scaling = 1000.0 if is_in_mm else 1.0
        constant_x = unit_scaling / self.fx
        constant_y = unit_scaling / self.fy

        # TODO: better sampling
        samples = \
            [
                (x + (w * 0.50), y + (h * 0.25)),  # top
                (x + (w * 0.25), y + (h * 0.50)),  # left
                (x + (w * 0.50), y + (h * 0.50)),  # center
                (x + (w * 0.75), y + (h * 0.50)),  # right
                (x + (w * 0.50), y + (h * 0.75)),  # bottom
            ]

        values = []
        for sample in samples:
            value = depth_image[int(sample[1]), int(sample[0])]
            values.append(value)

        depth = np.median(values)
        pose = PoseStamped()
        pose.pose.position.x = (float(x) - self.cx) * depth * constant_x
        pose.pose.position.y = (float(y) - self.cy) * depth * constant_y
        pose.pose.position.z = depth * unit_scaling
        pose.pose.orientation.w = 1
        pose.header.stamp = time_stamp
        pose.header.frame_id = self.camera_frame

        ts = rospy.Time.now()
        try:
            self.tf.waitForTransform(self.base_frame, self.camera_frame, time_stamp, rospy.Duration(4.0))
            transformed_pose = self.tf.transformPose(self.base_frame, pose)
            rospy.loginfo('timing transform: %r ' % (rospy.Time.now() - ts).to_sec())
            return transformed_pose
        except Exception, e:
            rospy.logerr("Exception %s" % e)
        return None

    @staticmethod
    def head_roi(image, person):
        parts = ['Nose', 'RightEar', 'RightEye', 'LeftEar', 'LeftEye']
        amount = int(sum([np.ceil(person[p]['confidence']) for p in parts]))
        if amount < 1:
            raise ValueError('body parts for face not found')
        vf = sum([np.ceil(person[p]['y']) for p in parts])
        v = int(np.floor(vf / amount))

        if amount <= 1:
            x = y = w = h = -1
            return image[y:y + h, x:x + w]

        dist_list_x = []
        if person['Nose']['x'] != 0:
            dist_list_x.append(np.abs(person['Nose']['x']))
        if person['RightEar']['x'] != 0:
            dist_list_x.append(np.abs(person['RightEar']['x']))
        if person['RightEye']['x'] != 0:
            dist_list_x.append(np.abs(person['RightEye']['x']))
        if person['LeftEar']['x'] != 0:
            dist_list_x.append(np.abs(person['LeftEar']['x']))
        if person['LeftEye']['x'] != 0:
            dist_list_x.append(np.abs(person['LeftEye']['x']))

        max_dist_u = np.amax(dist_list_x)

        min_dist_u = np.amin(dist_list_x)

        x = min_dist_u
        w = max_dist_u - min_dist_u
        h = w * 1.5
        y = v - h / 2

        if x + w >= image.shape[1]:
            w = w - np.abs((x + w) - image.shape[1])
        if y + h >= image.shape[0]:
            h = h - np.abs((y + h) - image.shape[0])

        return image[int(y):int(y + h), int(x):int(x + w)], x, y, w, h

    @staticmethod
    def upper_body_roi(image, person):
        parts = ['LeftShoulder', 'RightShoulder', 'LeftHip', 'RightHip']
        amount = int(sum([np.ceil(person[p]['confidence']) for p in parts]))

        if amount <= 1:
            x = y = w = h = -1
            return image[y:y + h, x:x + w]

        if person['LeftShoulder']['confidence'] > 0 and person['RightShoulder']['confidence'] > 0:
            y = person['RightShoulder']['y']
            w = np.abs(person['LeftShoulder']['x'] - person['RightShoulder']['x'])
            if (person['RightShoulder']['x'] - person['LeftShoulder']['x']) < 0:
                x = person['RightShoulder']['x']
            else:
                x = person['LeftShoulder']['x']

            if person['RightHip']['confidence'] > 0:
                h = (person['RightHip']['y'] - y)
            elif person['LeftHip']['confidence'] > 0:
                h = (person['LeftHip']['y'] - y)
            else:
                rospy.logerr("no hip found")
                h = w
        else:
            if person['RightHip']['confidence'] > 0 and person['LeftHip']['confidence'] > 0:
                if (person['RightShoulder']['confidence'] > 0) ^ (person['LeftShoulder']['confidence'] > 0):
                    w = np.abs(person['LeftHip']['x'] - person['RightHip']['x'])
                    y = person['LeftShoulder']['y'] + person['RightShoulder']['y']
                    if (person['RightHip']['x'] - person['LeftHip']['x']) < 0:
                        x = person['RightHip']['x']
                        h = person['RightHip']['y'] - person['LeftShoulder']['x'] - person['RightShoulder'][
                            'x']  # one of the shoulders values will be 0.
                    else:
                        x = person['LeftHip']['x']
                        h = person['LeftHip']['y'] - person['LeftShoulder']['x'] - person['RightShoulder']['x']
                else:
                    w = np.abs(person['LeftHip']['x'] - person['RightHip']['x'])
                    if (person['RightHip']['x'] - person['LeftHip']['x']) < 0:
                        x = person['RightHip']['x']
                        h = w
                        y = person['RightHip']['y'] - h
                    else:
                        x = person['LeftHip']['x']
                        h = w
                        y = person['LeftHip']['y'] - h
            else:
                if person['RightHip']['confidence'] > 0:
                    if person['RightShoulder']['confidence'] > 0:
                        x = person['RightShoulder']['x']
                        y = person['RightShoulder']['y']
                        h = np.abs(person['RightShoulder']['y'] - person['RightHip']['y'])
                        w = h * 0.5

                    if person['LeftShoulder']['confidence'] > 0:
                        x = person['RightHip']['x']
                        y = person['LeftShoulder']['y']
                        h = np.abs(person['LeftShoulder']['y'] - person['RightHip']['y'])
                        w = np.abs(person['LeftShoulder']['x'] - person['RightHip']['x'])
                if person['LeftHip']['confidence'] > 0:
                    if person['RightShoulder']['confidence'] > 0:
                        x = person['RightShoulder']['x']
                        y = person['RightShoulder']['y']
                        h = np.abs(person['RightShoulder']['y'] - person['LeftHip']['y'])
                        w = h * 0.5

                    if person['LeftShoulder']['confidence'] > 0:
                        y = person['LeftShoulder']['y']
                        h = np.abs(person['LeftShoulder']['y'] - person['LeftHip']['y'])
                        w = h * 0.5
                        x = person['LeftShoulder']['x'] - w
                else:
                    rospy.loginfo("No BB possible: RShoulder['confidence'] %f, LShoulder['confidence'] %f,"
                                  " RHip['confidence'] %f, LHip['confidence'] %f \n",
                                  person['RightShoulder']['confidence'], person['LeftShoulder']['confidence'],
                                  person['RightHip']['confidence'],
                                  person['LeftHip']['confidence'])

        if x + w >= image.shape[1]:
            w = w - np.abs((x + w) - image.shape[0])
        if y + h >= image.shape[0]:
            h = h - np.abs((y + h) - image.shape[1])

        if (w <= 0) or (h <= 0) or (x <= 0) or (y <= 0):
            rospy.loginfo("w or h <= 0")
            x = y = w = h = 0

        return image[int(y):int(y + h), int(x):int(x + w)], x, y, w, h

    @staticmethod
    def calcAngle(bodypart_one, bodypart_two):
        return np.abs(np.arctan2(bodypart_one['y'] - bodypart_two['y'], bodypart_one['x'] - bodypart_two['x']) * 180 / np.pi)


    @staticmethod
    def get_posture_and_gestures(person):
        SITTINGPERCENT = 0.4

        gestures = []
        vertical = 90
        horizontal = 180

        LShoulderLHipDist = np.sqrt(pow(person['LeftShoulder']['y'] - person['LeftHip']['y'], 2))
        LShoulderLWristAngle = Helper.calcAngle(person['LeftShoulder'], person['LeftWrist'])
        LShoulderLHipAngle = Helper.calcAngle(person['LeftShoulder'], person['LeftHip'])
        LKneeLHipDist = np.sqrt(pow(person['Leftknee']['y'] - person['LeftHip']['y'], 2))
        LAnkleLHipDist = np.sqrt(pow(person['LeftAnkle']['y'] - person['LeftHip']['y'], 2))

        RShoulderRHipDist = np.sqrt(pow(person['RightShoulder']['y'] - person['RightHip']['y'], 2))
        RShoulderRWristAngle = Helper.calcAngle(person['RightShoulder'], person['RightWrist'])
        RShoulderRHipAngle = Helper.calcAngle(person['RightShoulder'], person['RightHip'])
        RKneeRHipDist = np.sqrt(pow(person['RightKnee']['y'] - person['RightHip']['y'], 2))
        RAnkleRHipDist = np.sqrt(pow(person['RightAnkle']['y'] - person['RightHip']['y'], 2))

        if (((LKneeLHipDist < (LAnkleLHipDist * SITTINGPERCENT) or RKneeRHipDist < (RAnkleRHipDist * SITTINGPERCENT))
             and LKneeLHipDist > 0 and RKneeRHipDist > 0 and person['LeftAnkle']['Confidence'] > 0 and person['RightAnkle']['Confidence'] > 0)
                 or ((LKneeLHipDist < (LShoulderLHipDist * SITTINGPERCENT) or RKneeRHipDist < (RShoulderRHipDist * SITTINGPERCENT))
                 and person['LeftHip']['Confidence'] > 0 and person['RightHip']['Confidence'] > 0
                 and person['LeftKnee']['Confidence'] > 0 and person['RightKnee']['Confidence'] > 0
                 and person['LeftShoulder']['Confidence'] > 0 and person['RightShoulder']['Confidence'] > 0)):
            posture = Gesture.SITTING
        elif (np.abs(LShoulderLHipAngle - horizontal) < np.abs(LShoulderLHipAngle - vertical) or
              np.abs(RShoulderRHipAngle - horizontal) < np.abs(RShoulderRHipAngle - vertical) or
                LShoulderLHipAngle < 45 or RShoulderRHipAngle < 45):
            posture = Posture.LYING
        else:
            posture = Posture.STANDING
        
        if ((0 <= RShoulderRWristAngle and RShoulderRWristAngle <= 15) or (165 <= RShoulderRWristAngle and RShoulderRWristAngle <= 180)):
            gestures.append(Gesture.POINTING_RIGHT)
        elif ( ( 0 <= LShoulderLWristAngle and LShoulderLWristAngle <= 15 ) or ( 165 <= LShoulderLWristAngle and LShoulderLWristAngle <= 180 ) ):
            gestures.append(Gesture.POINTING_LEFT)
        elif (person['LeftElbow']['y'] < person['LeftShoulder']['y'] and person['LeftElbow']['y'] > 0 and person['LeftShoulder']['y'] > 0):
            gestures.append(Gesture.RAISING_LEFT_ARM)
        elif (person['RightElbow']['y'] < person['RightShoulder']['y'] and person['RightElbow']['y'] > 0 and person['RightShoulder']['y'] > 0):
            gestures.append(Gesture.RAISING_RIGHT_ARM)
        elif ((person['LeftWrist']['y'] < person['LeftEar']['y'] and person['LeftWrist']['y'] > 0 and person['LeftEar']['y'] > 0) or
                (person['RightWrist']['y'] < person['RightEar']['y'] and person['RightWrist']['y'] > 0 and person['RightEar']['y'] > 0)):
            gestures.push_back(Gesture.WAVING)
        else: 
            gestures.append(Gesture.NEUTRAL)
        
        return {'posture': posture, 'gestures': gestures}


class PoseEstimator:
    def __init__(self, cv_bridge, face_id=None, gender_age=None, resolution='208x192', model='mobilenet_thin'):

        model = rospy.get_param('~model', model)
        resolution = rospy.get_param('~resolution', resolution)  # old: '432x368'
        self.resize_out_ratio = float(rospy.get_param('~resize_out_ratio', '4.0'))
        self.face_id = face_id
        self.gender_age = gender_age
        self.cv_bridge = cv_bridge
        self.tf_lock = Lock()
        self.helper = Helper()
        self.result_pub = rospy.Publisher('/tf_pose/result', Image, queue_size=1)

        try:
            w, h = model_wh(resolution)
            graph_path = get_graph_path(model)
            rospack = rospkg.RosPack()
            graph_path = os.path.join(rospack.get_path('tfpose_ros'), graph_path)
        except Exception as e:
            rospy.logerr('invalid model: %s, e=%s' % (model, e))
            sys.exit(-1)

        self.pose_estimator = TfPoseEstimator(graph_path, target_size=(w, h))

    def get_person_attributes(self, color, depth, is_in_mm, do_gender_age=True, do_face_id=True):

        w = color.shape[1]
        h = color.shape[0]
        acquired = self.tf_lock.acquire(False)
        if not acquired:
            return

        try:
            time_stamp = rospy.Time.now()
            result = self.pose_estimator.inference(color, resize_to_default=True,
                                                   upsample_size=self.resize_out_ratio)
            humans = self.humans_to_dict(result, w, h)
            rospy.loginfo('timing tf_pose: %r' % (rospy.Time.now() - time_stamp).to_sec())
        finally:
            self.tf_lock.release()

        persons = []
        faces = []

        res_img = self.pose_estimator.draw_humans(color, result, imgcopy=True)
        self.result_pub.publish(self.cv_bridge.cv2_to_imgmsg(res_img, "bgr8"))

        for human in humans:
            person = PersonAttributesWithPose()

            pg = Helper.get_posture_and_gestures(human)
            person.attributes.posture = pg['posture']
            person.attributes.gestures = pg['gestures']

            try:
                ts = rospy.Time.now()
                f_roi, fx, fy, fw, fh = Helper.head_roi(color, human)
                rospy.loginfo('timing head_roi: %r' % (rospy.Time.now() - ts).to_sec())
                ts = rospy.Time.now()
                person.head_pose_stamped = self.helper.depth_lookup(color, depth, fx, fy, fw, fh, time_stamp, is_in_mm)
                rospy.loginfo('timing DLU: %r' % (rospy.Time.now() - ts).to_sec())
                face = self.cv_bridge.cv2_to_imgmsg(f_roi, "bgr8")
                faces.append(face)
                if do_face_id and self.face_id is not None and self.face_id.initialized:
                    ts = rospy.Time.now()
                    person.attributes.name = self.face_id.get_name(face)
                    rospy.loginfo('timing face_id: %r' % (rospy.Time.now() - ts).to_sec())
            except ValueError as e:
                rospy.loginfo('no face_roi found. added empty image to faces')
                faces.append(Image())
            try:

                ts = rospy.Time.now()
                b_roi, bx, by, bw, bh = Helper.upper_body_roi(color, human)
                rospy.loginfo('timing body_roi: %r' % (rospy.Time.now() - ts).to_sec())
                ts = rospy.Time.now()
                person.attributes.shirtcolor = ShirtColor.get_shirt_color(b_roi)
                rospy.loginfo('timing shirt_color: %r' % (rospy.Time.now() - ts).to_sec())
                person.pose_stamped = self.helper.depth_lookup(color, depth, bx, by, bw, bh, ts, is_in_mm)
            except ValueError as e:
                pass

            persons.append(person)

        if do_gender_age and self.gender_age is not None and self.gender_age.initialized:
            ts = rospy.Time.now()
            g_a = self.gender_age.get_genders_and_ages(faces)
            rospy.loginfo('timing gender_and_age: %r' % (rospy.Time.now() - ts).to_sec())

            for i in range(0, len(persons)):
                persons[i].attributes.gender_hyp = g_a[i].gender_probability
                persons[i].attributes.age_hyp = g_a[i].age_probability

        # rospy.loginfo(persons)
        return persons

    def get_closest_person(self, persons, depth):
        # TODO
        pass

    def get_closest_person_face(self, color, depth, is_in_mm):
        w = color.shape[1]
        h = color.shape[0]
        persons = self.humans_to_dict(self.pose_estimator.inference(color, resize_to_default=True,
                                                                    upsample_size=self.resize_out_ratio), w, h)
        # TODO: filter closest
        return self.helper.head_roi(color, persons[0])[0]

    def get_closest_person_body_roi(self, color, depth, is_in_mm):
        w = color.shape[1]
        h = color.shape[0]
        persons = self.humans_to_dict(self.pose_estimator.inference(color, resize_to_default=True,
                                                                    upsample_size=self.resize_out_ratio), w, h)
        # TODO: filter closest
        body_roi = RegionOfInterest()
        try:
            roi = self.helper.upper_body_roi(color, persons[0])
            body_roi.x_offset = roi[1]
            body_roi.y_offset = roi[2]
            body_roi.width = roi[3]
            body_roi.height = roi[4]
        except Exception as e:
            rospy.logerr('error while getting shirt roi: %s' % e)
        rospy.loginfo('shirt roi: %r' % body_roi)
        return body_roi

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
