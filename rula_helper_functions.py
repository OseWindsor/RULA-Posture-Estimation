import time
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from numpy import expand_dims, asarray
from PIL import Image

from person_detector_model import make_yolov3_model, WeightReader
from rula_model import posture_estimation_model_func

body_model_img_size_width = 128
body_model_img_size_height = 128
body_channel = 1
classes_count = 17
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the probability threshold for detected objects
class_threshold = 0.75

# define the class labels
labels = ["person"]

rula_repo_table_path = 'static/resource/RULA_repository_table.xlsx'
rula_group_a_df = pd.read_excel(rula_repo_table_path, 'group_A_table', header=None)
rula_group_b_df = pd.read_excel(rula_repo_table_path, 'group_B_table', header=None)
rula_grand_score_df = pd.read_excel(rula_repo_table_path, 'grand_score_table1', header=None)

# Initialize the body key points
# left_knee_keypoint_coordinates = []
# right_knee_keypoint_coordinates = []
# left_ankle_keypoint_coordinates = []
# right_ankle_keypoint_coordinates = []
# right_trunk_keypoint_coordinates = []
# left_trunk_keypoint_coordinates = []
# right_shoulder_keypoint_coordinates = []
# left_shoulder_keypoint_coordinates = []
# left_elbow_keypoint_coordinates = []
# right_elbow_keypoint_coordinates = []
# right_wrist_keypoint_coordinates = []
# left_wrist_keypoint_coordinates = []
# right_knuckle_keypoint_coordinates = []
# left_knuckle_keypoint_coordinates = []
# left_eye_keypoint_coordinates = []
# right_eye_keypoint_coordinates = []
# neck_keypoint_coordinates = []
# nose_keypoint_coordinates = []
#
# head_keypoint_coordinates = []

is_rested = True
neck_twist_thres = 15
neck_side_bend_thres = 8
trunk_twist_thres = 6
trunk_side_bend_thres = 15
wrist_bend_thres = 15

rula_result_dict = {}

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.get_score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if objectness.all() <= obj_thresh: continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    data = cv2.resize(data, (224, 224))
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='yellow', linewidth='2')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='yellow')
    # show the plot
    pyplot.show()


def load_image_pixels(filename, shape):
    # load image to get its shape
    image = load_img(filename)
    image = image.resize((224, 224))
    # width, height = image.size
    width, height = 224, 224
    # load image with required size
    image = load_img(filename, target_size=shape)
    image = img_to_array(image)

    # grayscale image normalization
    image = image.astype('float32')
    image /= 255.0

    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

def read_classes(classes_path):
  with open(classes_path) as f:
    class_names = f.readlines()
  # stripping by default for space
  class_names = [c.strip() for c in class_names]
  return class_names

def extract_person_from_image(src_img_input, required_size=(body_model_img_size_width, body_model_img_size_height)):
    print('Detecting person in image: ' + str(src_img_input))
    person_complete_data = []
    src_image_read = plt.imread(src_img_input)
    src_image_read = cv2.resize(src_image_read, (224, 224))

    image, image_w, image_h = load_image_pixels(src_img_input, (224, 224))
    # make prediction
    yhat = yolov3.predict(image)
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, 224, 224)

    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, 224, 224)

    # suppress non-maximal boxes
    do_nms(boxes, 0.5)

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    # summarize what we found
    boundary_threshold = 0.10
    print('No of person found: ' + str(len(v_boxes)))
    if 0 < len(v_boxes) < 5:

        if len(src_image_read.shape) == 3:
            # print('converting the cropped image to gray')
            src_image_read = cv2.cvtColor(src_image_read, cv2.COLOR_RGB2GRAY)

        for i in range(len(v_boxes)):
            person = []
            # print(v_labels[i], v_scores[i])
            box = v_boxes[i]
            # get coordinates
            x1, y1, x2, y2 = box.xmin, box.ymin, box.xmax, box.ymax
            boundary_threshold_x = int(abs(x2 - x1) * boundary_threshold)
            boundary_threshold_y = int(abs(y2 - y1) * boundary_threshold)
            x1 = int(x1 - boundary_threshold_x)
            x2 = int(x2 + boundary_threshold_x)
            y1 = int(y1 - boundary_threshold_y)
            y2 = int(y2 + boundary_threshold_y)
            # avoid beyond tile error
            x1_lim = 0 if x1 < 0 else x1
            y1_lim = 0 if y1 < 0 else y1
            x2_lim = 224 if x2 > 224 else x2
            y2_lim = 224 if y2 > 224 else y2

            # print('person boundary: ')
            # print(x1_lim, y1_lim, x2_lim, y2_lim)

            person.append([x1_lim, y1_lim, x2_lim, y2_lim])
            # extract the person in grayscale format
            person_boundary = src_image_read[y1_lim:y2_lim, x1_lim:x2_lim]

            # cv2_imshow(person_boundary)
            person_image = Image.fromarray(person_boundary)
            # resize pixels to the model size
            person_image = person_image.resize(required_size)
            person_array = asarray(person_image, dtype=float)
            person_complete_data.append([person_array, person])

    return person_complete_data


def pre_process_body_image(src_body_img_arr, mode='train'):
    # print('Preprocessing image')
    # image standardization
    # image_norm = img_standardization(src_body_img_arr)
    image_norm = src_body_img_arr / 255.0
    return image_norm


def predict_model(model_name, test_data):
    print('Predicting')
    start_time = time.time()
    prediction = model_name.predict(test_data)
    elapsed_time = time.time() - start_time
    print('Time taken to predict the result: ' + str(elapsed_time) + ' seconds')
    print(prediction)
    return prediction


# trunk,shoulder & shoulder,Elbow
def calc_upper_arm_angle(point_a, base_point, point_b):
    trunk_joint_coordinate = np.array([point_a[0], point_a[1]])
    shoulder_joint_coordinate = np.array([base_point[0], base_point[1]])
    elbow_joint_coordinate = np.array([point_b[0], point_b[1]])

    ba = trunk_joint_coordinate - shoulder_joint_coordinate
    bc = elbow_joint_coordinate - shoulder_joint_coordinate

    # positive if elbow is behind shoulder (Anti-clockwise) i.e extension
    # negative: flexion
    upper_arm_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))

    return upper_arm_angle

def calc_upper_arm_abduction(point_a, base_point, point_b):
    trunk_joint_coordinate = np.array([point_a[0], point_a[1]])
    shoulder_joint_coordinate = np.array([base_point[0], base_point[1]])
    elbow_joint_coordinate = np.array([point_b[0], point_b[1]])

    ba = trunk_joint_coordinate - shoulder_joint_coordinate
    bc = elbow_joint_coordinate - shoulder_joint_coordinate
    upper_arm_angle = abs(np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc))))
    print(upper_arm_angle)
    return upper_arm_angle

# Extend the line connecting shoulder and elbow and find angle between upper arm and lower arm(elbow, wrist)
# Shoulder[502.5,166.0]; elbow[502, 267]; wrist[542.5,345.5]
def calc_lower_arm_angle(input_shoulder_joint_coordinate, input_elbow_joint_coordinate, input_wrist_joint_coordinate):
    point_c = np.array([input_shoulder_joint_coordinate[0], input_shoulder_joint_coordinate[1]])
    base_point = np.array([input_elbow_joint_coordinate[0], input_elbow_joint_coordinate[1]])
    point_b = np.array([input_wrist_joint_coordinate[0], input_wrist_joint_coordinate[1]])

    ba = point_b - base_point
    bc = point_c - base_point

    lower_arm_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print('Func lower arm angle: ' + str(lower_arm_angle))
    if lower_arm_angle<0:
        lower_arm_angle = 180 - abs(lower_arm_angle)
    else:
        lower_arm_angle = 0
    print(lower_arm_angle)
    return lower_arm_angle


def calc_lower_arm_work_midline(right_shoulder_joint_coordinate, left_shoulder_joint_coordinate,
                                right_wrist_coordinate, left_wrist_coordinate):
    is_lower_arm_working_midline = False

    if right_shoulder_joint_coordinate[0] - left_shoulder_joint_coordinate[0] > 0:
        print('front view with left arm towards left side of image')
        if left_wrist_coordinate[0] > right_wrist_coordinate[0]:
            is_lower_arm_working_midline = False
        else:
            is_lower_arm_working_midline = True

    elif left_shoulder_joint_coordinate[0] - right_shoulder_joint_coordinate[0] > 0:
        print('front view with right arm towards left side of image')
        if right_wrist_coordinate[0] > left_wrist_coordinate[0]:
            is_lower_arm_working_midline = True
        else:
            is_lower_arm_working_midline = False
    print(is_lower_arm_working_midline)
    return is_lower_arm_working_midline


# line between elbow and wrist extended and interior angle is calculated between lower arm and wrist(wrist, knuckle)
def calc_wrist_posture_angle(elbow_joint_coordinate, wrist_joint_coordinate, knuckle_joint_coordinate):
    wrist_status = False
    point_c = np.array([elbow_joint_coordinate[0], elbow_joint_coordinate[1]])
    base_point = np.array([wrist_joint_coordinate[0], wrist_joint_coordinate[1]])
    point_b = np.array([knuckle_joint_coordinate[0], knuckle_joint_coordinate[1]])
    ba = point_b - base_point
    bc = point_c - base_point

    wrist_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print('Func wrist angle: ' + str(wrist_angle))

    if wrist_angle == 180:
        print('wrist neutral')
        wrist_status = False
        wrist_angle = 180 - wrist_angle
    elif -180 < wrist_angle < 0:
        # clockwise
        print('wrist extension')
        wrist_status = False
    elif 180 > wrist_angle > 0:
        # anticlockwise
        print('wrist flexion')
        wrist_status = True

    wrist_angle = 180-abs(wrist_angle)

    return wrist_status, wrist_angle


# +: Flexion, -: Extension
def calc_neck_posture_angle(trunk_joint_coordinate, neck_joint_coordinate, head_joint_coordinate):
    neck_status = True
    point_c = np.array([trunk_joint_coordinate[0], trunk_joint_coordinate[1]])
    base_point = np.array([neck_joint_coordinate[0], neck_joint_coordinate[1]])
    point_b = np.array([head_joint_coordinate[0], head_joint_coordinate[1]])

    ba = point_b - base_point
    bc = point_c - base_point

    # positive if head is behind neck i.e extension
    neck_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print('Func neck angle: ' + str(neck_angle))
    if neck_angle==180:
        neck_status = True
        print('Neck neutral')
        neck_angle = 180 - neck_angle
    elif 180 > neck_angle > 0:
        neck_status = True
        print('Neck flexion')
    elif -180 < neck_angle < 0:
        neck_status = True
        print('Neck Extension')
    neck_angle = 180-abs(neck_angle)
    print('Neck angle: ' + str(neck_angle))

    if not neck_status:
        if neck_angle<20:
            neck_angle = 0
            neck_status = True
    elif neck_angle !=0 and 10<neck_angle:
        neck_angle = abs(neck_angle - 15)

    return neck_status, neck_angle


# +: Flexion, -: extension
def calc_trunk_posture_angle(vertical_joint_coordinate, trunk_joint_coordinate, neck_joint_coordinate):
    point_b = np.array([vertical_joint_coordinate[0], vertical_joint_coordinate[1]])
    base_point = np.array([trunk_joint_coordinate[0], trunk_joint_coordinate[1]])
    point_c = np.array([neck_joint_coordinate[0], neck_joint_coordinate[1]])

    ba = point_b - base_point
    bc = point_c - base_point

    trunk_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))

    print('Func trunk angle: ' + str(trunk_angle))

    trunk_angle = abs(trunk_angle)

    return trunk_angle


def calc_upper_arm_score(input_upper_arm_angle):
    upper_arm_score = 1

    if input_upper_arm_angle > 0:
        print('Upper arm Extension')
        _is_upper_arm_flexion = False
        if 0 < input_upper_arm_angle <= 20:
            upper_arm_score = 1
        elif 20 < input_upper_arm_angle:
            upper_arm_score = 2
    else:
        _is_upper_arm_flexion = True
        print('Upper arm Flexion')
        input_upper_arm_angle = abs(input_upper_arm_angle)
        if 0 < input_upper_arm_angle <= 20:
            upper_arm_score = 1
        elif 20 < input_upper_arm_angle <= 45:
            upper_arm_score = 2
        elif 45 < input_upper_arm_angle <= 90:
            upper_arm_score = 3
        elif 90 < input_upper_arm_angle:
            upper_arm_score = 4

    return _is_upper_arm_flexion, upper_arm_score


def calc_lower_arm_score(input_lower_arm_angle):
    lower_arm_score = 0

    if input_lower_arm_angle < 0:
        _is_lower_arm_flexion = False
        print('Lower arm extension not possible')

    else:
        print('Lower arm flexion')
        _is_lower_arm_flexion = True
        if 60 < input_lower_arm_angle < 100:
            lower_arm_score = 1
        elif 0<= input_lower_arm_angle <= 60:
            lower_arm_score = 2
        elif input_lower_arm_angle >= 100:
            lower_arm_score = 2

    return _is_lower_arm_flexion, lower_arm_score


def calc_wrist_score(input_wrist_angle):
    wrist_posture_score = 1
    if input_wrist_angle == 0:
        wrist_posture_score = 1
    elif 0 < input_wrist_angle <= 15:
        wrist_posture_score = 2
    elif input_wrist_angle > 15:
        wrist_posture_score = 3

    return wrist_posture_score


def calc_neck_score(input_neck_angle, input_neck_status):
    # Flexion or neutral
    neck_posture_score = 1
    if input_neck_status:
        if 0 <= input_neck_angle < 10:
            neck_posture_score = 1
        elif 10 <= input_neck_angle < 20:
            neck_posture_score = 2
        elif input_neck_angle >= 20:
            neck_posture_score = 3
    else:
        neck_posture_score = 4

    return neck_posture_score

# distance between nose and both shoulders must be same
def calc_neck_twist(left_shoulder_coordinate_pos, right_shoulder_coordinate_pos,
                    nose_coordinate_pos):
    right_shoulder_joint_coordinate = np.array([right_shoulder_coordinate_pos[0], right_shoulder_coordinate_pos[1]])
    left_shoulder_joint_coordinate = np.array([left_shoulder_coordinate_pos[0], left_shoulder_coordinate_pos[1]])
    nose_joint_coordinate = np.array([nose_coordinate_pos[0], nose_coordinate_pos[1]])

    dist_1 = int(math.sqrt(((nose_joint_coordinate[0] - right_shoulder_joint_coordinate[0]) ** 2) + (
                (nose_joint_coordinate[1] - right_shoulder_joint_coordinate[1]) ** 2)))
    dist_2 = int(math.sqrt(((nose_joint_coordinate[0] - left_shoulder_joint_coordinate[0]) ** 2) + (
                (nose_joint_coordinate[1] - left_shoulder_joint_coordinate[1]) ** 2)))

    print(dist_1, dist_2)

    if abs(dist_1 - dist_2) > neck_twist_thres:
        _is_neck_side_twist_pos = True
    else:
        _is_neck_side_twist_pos = False

    return _is_neck_side_twist_pos


# angle of head from mid trunk and mid shoulder
def calc_neck_side_bending(trunk_coordinate_pos, neck_coordinate_pos, head_coordinate_pos):
    point_c = np.array([trunk_coordinate_pos[0], trunk_coordinate_pos[1]])
    base_point = np.array([neck_coordinate_pos[0], neck_coordinate_pos[1]])
    point_b = np.array([head_coordinate_pos[0], head_coordinate_pos[1]])

    ba = point_b - base_point
    bc = point_c - base_point

    neck_bend_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print('Func neck side bent angle: ' + str(neck_bend_angle))

    neck_bend_angle = abs(neck_bend_angle)

    if neck_bend_angle <= 180:
        neck_bend_angle = 180 - neck_bend_angle

    if neck_bend_angle >= 5:
        _is_neck_side_bent_pos = True
    else:
        _is_neck_side_bent_pos = False

    return _is_neck_side_bent_pos


def calc_trunk_score(input_trunk_angle):
    trunk_posture_score = 1
    if input_trunk_angle == 0:
        print('Neutral trunk position')
        trunk_posture_score = 1
    elif 0 < input_trunk_angle <= 20:
        print('Trunk flexion')
        trunk_posture_score = 2
    elif 20 < input_trunk_angle <= 60:
        print('Trunk flexion')
        trunk_posture_score = 3
    elif input_trunk_angle > 60:
        print('Trunk flexion')
        trunk_posture_score = 4

    return trunk_posture_score


# difference of length of opposite pairs of trunk and shoulder must be less than 10
def calc_trunk_twist(left_shoulder_coordinate_pos, right_shoulder_coordinate_pos,
                     left_trunk_coordinate_pos, right_trunk_coordinate_pos,
                     trunk_side_bend_status):
    right_shoulder_joint_coordinate = np.array([right_shoulder_coordinate_pos[0], right_shoulder_coordinate_pos[1]])
    left_shoulder_joint_coordinate = np.array([left_shoulder_coordinate_pos[0], left_shoulder_coordinate_pos[1]])
    right_trunk_joint_coordinate = np.array([right_trunk_coordinate_pos[0], right_trunk_coordinate_pos[1]])
    left_trunk_joint_coordinate = np.array([left_trunk_coordinate_pos[0], left_trunk_coordinate_pos[1]])

    dist_1 = int(math.sqrt(((left_trunk_joint_coordinate[0] - right_shoulder_joint_coordinate[0]) ** 2) +
                            ((left_trunk_joint_coordinate[1] - right_shoulder_joint_coordinate[1]) ** 2)))
    dist_2 = int(math.sqrt(((right_trunk_joint_coordinate[0] - left_shoulder_joint_coordinate[0]) ** 2) +
                            ((right_trunk_joint_coordinate[1] - left_shoulder_joint_coordinate[1]) ** 2)))

    print(dist_1, dist_2)

    if abs(dist_1 - dist_2) > trunk_twist_thres:
        _is_trunk_side_twist_pos = True
        if trunk_side_bend_status:
            print('Trunk side twist result can not be relied because of side bent')
            if abs(dist_1 - dist_2) > 10:
                print('Strong probability of trunk twisting')
                _is_trunk_side_twist_pos = True
            else:
                _is_trunk_side_twist_pos = False
    else:
        _is_trunk_side_twist_pos = False


    return _is_trunk_side_twist_pos


def calc_wrist_bent(elbow_coordinate_pos, wrist_coordinate_pos, knuckle_coordinate_pos, img_view):
    point_c = np.array([elbow_coordinate_pos[0], elbow_coordinate_pos[1]])
    base_point = np.array([wrist_coordinate_pos[0], wrist_coordinate_pos[1]])
    point_b = np.array([knuckle_coordinate_pos[0], knuckle_coordinate_pos[1]])
    ba = point_b - base_point
    bc = point_c - base_point

    wrist_bent_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print('Func wrist bent angle: ' + str(wrist_bent_angle))

    wrist_bent_angle = abs(wrist_bent_angle)

    if wrist_bent_angle <= 180:
        wrist_bent_angle = 180 - wrist_bent_angle

    wrist_bent_status = False
    print('Wrist bent angle: ' + str(wrist_bent_angle))
    if img_view == 'front':
        if wrist_bent_angle > wrist_bend_thres:
            wrist_bent_status = True
        else:
            wrist_bent_status = False
    else:
        print('Not an appropriate view. Needs front view')
        wrist_bent_status = False
    return wrist_bent_status


def calc_trunk_side_bent(left_shoulder_coordinate_pos, right_shoulder_coordinate_pos):
    trunk_side_bent_angle = 180

    right_shoulder_joint_coordinate = np.array([right_shoulder_coordinate_pos[0], right_shoulder_coordinate_pos[1]])
    left_shoulder_joint_coordinate = np.array([left_shoulder_coordinate_pos[0], left_shoulder_coordinate_pos[1]])
    point_c = np.array([right_shoulder_coordinate_pos[0] * 0.5, right_shoulder_coordinate_pos[1]])

    ba = left_shoulder_joint_coordinate - right_shoulder_joint_coordinate
    bc = point_c - right_shoulder_joint_coordinate

    trunk_side_bent_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print('Func trunk side bend angle: ' + str(trunk_side_bent_angle))

    trunk_side_bent_angle = 180 - abs(trunk_side_bent_angle)

    if trunk_side_bent_angle >= trunk_side_bend_thres:
        is_trunk_side_bent_pos = True
    else:
        is_trunk_side_bent_pos = False

    print('Trunk side bend angle: ' + str(trunk_side_bent_angle))
    return is_trunk_side_bent_pos


# difference of length of right and left legs(ankle to knee) must be less than 10
def calc_leg_support(left_ankle_coordinate_pos, right_ankle_coordinate_pos, left_trunk_coordinate_pos,
                     right_trunk_coordinate_pos):
    right_ankle_joint_coordinate = np.array([right_ankle_coordinate_pos[0], right_ankle_coordinate_pos[1]])
    left_ankle_joint_coordinate = np.array([left_ankle_coordinate_pos[0], left_ankle_coordinate_pos[1]])
    right_trunk_joint_coordinate = np.array([left_trunk_coordinate_pos[0], left_trunk_coordinate_pos[1]])
    left_trunk_joint_coordinate = np.array([right_trunk_coordinate_pos[0], right_trunk_coordinate_pos[1]])

    dist_1 = left_ankle_joint_coordinate[1]
    dist_2 = right_ankle_joint_coordinate[1]

    if abs(dist_1 - dist_2) > abs(right_trunk_joint_coordinate[0]-left_trunk_joint_coordinate[0]):
        _is_leg_supported_pos = False
    else:
        _is_leg_supported_pos = True

    return _is_leg_supported_pos

def find_rula_posture_score_a(upper_arm_score, lower_arm_score, wrist_posture_score, wrist_twist_score):
  upper_posture_score = 0

  upper_arm_df = rula_group_a_df.loc[(rula_group_a_df[0] == upper_arm_score) | (rula_group_a_df[0] == 0)]

  lower_arm_df = upper_arm_df.loc[(rula_group_a_df[1] == lower_arm_score) | (rula_group_a_df[0] == 0)]

  lower_arm_df = lower_arm_df.reset_index(drop = True)
  for col in range(lower_arm_df.shape[1]):
    if lower_arm_df[col][0] == wrist_posture_score:
      if lower_arm_df[col][1] == wrist_twist_score:
        upper_posture_score = lower_arm_df[col][2]
  print('Score A: '+str(upper_posture_score))
  return upper_posture_score


def find_rula_posture_score_b(neck_posture_score, trunk_posture_score, leg_posture_score):
  lower_posture_score = 0
  neck_df = rula_group_b_df.loc[(rula_group_b_df[0] == neck_posture_score) | (rula_group_b_df[0] == 0)]

  neck_df = neck_df.reset_index(drop=True)
  for col in range(neck_df.shape[1]):
    if neck_df[col][0] == trunk_posture_score:
      if neck_df[col][1] == leg_posture_score:
        lower_posture_score = neck_df[col][2]

  print('Score B: '+str(lower_posture_score))
  return lower_posture_score


def find_rula_grand_score(upper_body_posture_score, lower_body_posture_score):
  # Assumption
  upper_body_posture_score = upper_body_posture_score+2
  lower_body_posture_score = lower_body_posture_score+2
  if upper_body_posture_score>6:
      upper_body_posture_score = 6
  if lower_body_posture_score > 7:
      lower_body_posture_score = 7
  grand_score = 0
  for col in range(rula_grand_score_df.shape[1]):
    if rula_grand_score_df[col][0] == lower_body_posture_score:
      for row in range(rula_grand_score_df.shape[0]):
        if rula_grand_score_df[0][row] == upper_body_posture_score:
          grand_score = rula_grand_score_df[col][row]

  print('Grand score is: '+str(grand_score))
  return grand_score



def decode_pose_estimate_net(input_test_img_name_s1, input_extracted_test_body_s1, model_name):
    input_test_label_data_list_s1 = []
    pre_processed_test_image_s1 = pre_process_body_image(input_extracted_test_body_s1[0], mode='test')
    body_bbox_s1_x1 = input_extracted_test_body_s1[1][0][0]
    body_bbox_s1_y1 = input_extracted_test_body_s1[1][0][1]
    body_bbox_s1_x2 = input_extracted_test_body_s1[1][0][2]
    body_bbox_s1_y2 = input_extracted_test_body_s1[1][0][3]

    x_test_s1 = np.array(pre_processed_test_image_s1)
    x_test_s1 = np.array(x_test_s1).reshape([-1, body_model_img_size_width, body_model_img_size_height, body_channel])
    print(x_test_s1.shape)

    # predict keypoints in image
    prediction_s1 = predict_model(model_name, x_test_s1)

    test_src_img_s1 = cv2.imread(input_test_img_name_s1, cv2.IMREAD_COLOR)
    test_src_img_s1 = cv2.resize(test_src_img_s1, (224, 224))
    test_img_s1_height, test_img_s1_width, test_img_channel = test_src_img_s1.shape

    body_bbox_s1_width = int(abs(body_bbox_s1_x1 - body_bbox_s1_x2))
    body_bbox_s1_height = int(abs(body_bbox_s1_y1 - body_bbox_s1_y2))

    body_bbox_center_x1 = int(body_bbox_s1_x1 + (body_bbox_s1_width * 0.5))
    body_bbox_center_y1 = int(body_bbox_s1_y1 + (body_bbox_s1_height * 0.5))
    # calculate absolute coordinates of labels
    for test_i in range(0, classes_count):
        test_img_label_s1 = ''
        test_s1_x_coordinate = int(
            body_bbox_s1_x1 + ((prediction_s1[0][2 * test_i] * (224 / 128)) * (body_bbox_s1_width / 224)))
        test_s1_y_coordinate = int(
            body_bbox_s1_y1 + ((prediction_s1[0][(2 * test_i) + 1] * (224 / 128)) * (body_bbox_s1_height / 224)))

        for test_class_n, test_class_id in model_class_id_list.items():
            if test_class_id == test_i + 1:
                test_img_label_s1 = test_class_n
        input_test_label_data_list_s1.append([test_s1_x_coordinate, test_s1_y_coordinate, test_img_label_s1])

    # calculate neck coordinate
    input_test_label_data_list_s1.append(
        [abs(input_test_label_data_list_s1[0][0] + input_test_label_data_list_s1[1][0]) * 0.5,
         abs(input_test_label_data_list_s1[0][1] + input_test_label_data_list_s1[1][1]) * 0.5,
         'neck'])
    head_y = abs(input_test_label_data_list_s1[17][1] - input_test_label_data_list_s1[0][1])
    input_test_label_data_list_s1.append(
        [abs(input_test_label_data_list_s1[14][0] + input_test_label_data_list_s1[15][0]) * 0.5,
         (abs(input_test_label_data_list_s1[14][1] + input_test_label_data_list_s1[15][1]) * 0.5) - head_y,
         'head'])

    # Plot keypoints
    for test_label_data in input_test_label_data_list_s1:
        print('Plotting point ' + str(int(test_label_data[0])) + str(', ') + str(
            int(test_label_data[1])) + ' for label ' + str(test_label_data[2]))
        test_src_img_s1 = cv2.circle(img=test_src_img_s1, center=(int(test_label_data[0]), int(test_label_data[1])),
                                     radius=2,
                                     color=(255, 0, 0), thickness=-1)
    cv2.imshow('Test image',test_src_img_s1)
    return input_test_label_data_list_s1


def plot_estimate_rula_score(input_test_front_img, input_test_side_img, user_param):
    img_left_upper_arm_score = 1
    img_right_upper_arm_score = 1
    img_left_lower_arm_score = 1
    img_right_lower_arm_score = 1
    img_left_wrist_score = 1
    img_right_wrist_score = 1
    img_left_wrist_twist_score = 1
    img_right_wrist_twist_score = 1
    img_neck_score = 1
    img_trunk_score = 1
    img_leg_posture_score = 1

    img_left_upper_arm_angle = 0
    img_right_upper_arm_angle = 0

    right_posture_score_a = 1
    left_posture_score_a = 1
    posture_score_b = 1
    left_grand_posture_score = 1
    right_grand_posture_score = 1

    view = 'right'
    img_view = user_param[1]

    _is_operator_lean = user_param[0]
    # Fetch the side and front view image

    test_img_front = input_test_front_img
    test_img_side = input_test_side_img
    if img_view == 'R':
        view = 'right'
    elif img_view == 'L':
        view = 'left'
    extracted_test_bodies_side = extract_person_from_image(test_img_side)

    if len(extracted_test_bodies_side) == 1:
        print('No of people detected: ' + str(len(extracted_test_bodies_side)))
        for extracted_test_body_side in extracted_test_bodies_side:
            test_label_data_list_side = []
            test_label_data_list_side = decode_pose_estimate_net(test_img_side, extracted_test_body_side,
                                                                 posture_estimation_model)
            # continue
            body_keypoint_df = pd.DataFrame(test_label_data_list_side)
            body_keypoint_df = body_keypoint_df.set_index([2])
            left_shoulder_keypoint_df = body_keypoint_df.loc['left_shoulder']
            left_shoulder_keypoint_coordinates = [left_shoulder_keypoint_df[0], left_shoulder_keypoint_df[1]]
            right_shoulder_keypoint_df = body_keypoint_df.loc['right_shoulder']
            right_shoulder_keypoint_coordinates = [right_shoulder_keypoint_df[0], right_shoulder_keypoint_df[1]]

            neck_keypoint_df = body_keypoint_df.loc['neck']
            neck_keypoint_coordinates = [neck_keypoint_df[0], neck_keypoint_df[1]]
            nose_keypoint_df = body_keypoint_df.loc['nose']
            nose_keypoint_coordinates = [nose_keypoint_df[0], nose_keypoint_df[1]]

            head_keypoint_df = body_keypoint_df.loc['head']
            head_keypoint_coordinates = [head_keypoint_df[0], head_keypoint_df[1]]
            left_elbow_keypoint_df = body_keypoint_df.loc['left_elbow']
            left_elbow_keypoint_coordinates = [left_elbow_keypoint_df[0], left_elbow_keypoint_df[1]]
            right_elbow_keypoint_df = body_keypoint_df.loc['right_elbow']
            right_elbow_keypoint_coordinates = [right_elbow_keypoint_df[0], right_elbow_keypoint_df[1]]
            left_wrist_keypoint_df = body_keypoint_df.loc['left_wrist']
            left_wrist_keypoint_coordinates = [left_wrist_keypoint_df[0], left_wrist_keypoint_df[1]]
            right_wrist_keypoint_df = body_keypoint_df.loc['right_wrist']
            right_wrist_keypoint_coordinates = [right_wrist_keypoint_df[0], right_wrist_keypoint_df[1]]
            left_knuckle_keypoint_df = body_keypoint_df.loc['left_knuckle']
            left_knuckle_keypoint_coordinates = [left_knuckle_keypoint_df[0], left_knuckle_keypoint_df[1]]
            right_knuckle_keypoint_df = body_keypoint_df.loc['right_knuckle']
            right_knuckle_keypoint_coordinates = [right_knuckle_keypoint_df[0], right_knuckle_keypoint_df[1]]

            left_trunk_keypoint_df = body_keypoint_df.loc['left_trunk']
            left_trunk_keypoint_coordinates = [left_trunk_keypoint_df[0], left_trunk_keypoint_df[1]]
            right_trunk_keypoint_df = body_keypoint_df.loc['right_trunk']
            right_trunk_keypoint_coordinates = [right_trunk_keypoint_df[0], right_trunk_keypoint_df[1]]

            mid_trunk_keypoint_coordinates = [abs(left_trunk_keypoint_df[0] + right_trunk_keypoint_df[0]) * 0.5,
                                              abs(left_trunk_keypoint_df[1] + right_trunk_keypoint_df[1]) * 0.5]

            mid_vertical_keypoint_coordinates = [mid_trunk_keypoint_coordinates[0],
                                                 mid_trunk_keypoint_coordinates[1]*0.5]

            # Upper Arm Angle
            img_left_upper_arm_angle = calc_upper_arm_angle(left_trunk_keypoint_coordinates,
                                                            left_shoulder_keypoint_coordinates,
                                                            left_elbow_keypoint_coordinates)
            print('Left Upper arm angle: ' + str(img_left_upper_arm_angle))
            # status: true means flexion
            img_left_upper_arm_status, img_left_upper_arm_score = calc_upper_arm_score(img_left_upper_arm_angle)
            print('Corresponding Score: ' + str(img_left_upper_arm_score))
            img_right_upper_arm_angle = calc_upper_arm_angle(right_trunk_keypoint_coordinates,
                                                             right_shoulder_keypoint_coordinates,
                                                             right_elbow_keypoint_coordinates)
            print('Right Upper arm angle: ' + str(img_right_upper_arm_angle))

            img_right_upper_arm_status, img_right_upper_arm_score = calc_upper_arm_score(img_right_upper_arm_angle)
            print('Corresponding Score: ' + str(img_right_upper_arm_score))

            if _is_operator_lean:
                print('Operator leaned')
                img_left_upper_arm_score = img_left_upper_arm_score - 1
                img_right_upper_arm_score = img_right_upper_arm_score - 1
                print('Upper arm score updated due to lean posture: ' + str(img_left_upper_arm_score))
            else:
                _is_operator_lean = False
                print('Operator not leaned')

            # Lower Arm Posture
            img_left_lower_arm_angle = calc_lower_arm_angle(left_shoulder_keypoint_coordinates,
                                                            left_elbow_keypoint_coordinates,
                                                            left_wrist_keypoint_coordinates)
            print('Left Lower arm angle: ' + str(img_left_lower_arm_angle))
            left_lower_arm_status, img_left_lower_arm_score = calc_lower_arm_score(img_left_lower_arm_angle)
            print('Left lower arm Corresponding Score: ' + str(img_left_lower_arm_score))
            img_right_lower_arm_angle = calc_lower_arm_angle(right_shoulder_keypoint_coordinates,
                                                             right_elbow_keypoint_coordinates,
                                                             right_wrist_keypoint_coordinates)
            print('Right Lower arm angle: ' + str(img_right_lower_arm_angle))
            right_lower_arm_status, img_right_lower_arm_score = calc_lower_arm_score(img_right_lower_arm_angle)
            print('right lower arm Corresponding Score: ' + str(img_right_lower_arm_score))

            print('Left wrist posture')
            left_wrist_status, img_left_wrist_angle = calc_wrist_posture_angle(left_elbow_keypoint_coordinates,
                                                            left_wrist_keypoint_coordinates,
                                                            left_knuckle_keypoint_coordinates)
            print('Left wrist angle: ' + str(img_left_wrist_angle))
            img_left_wrist_score = calc_wrist_score(img_left_wrist_angle)
            print('Left wrist Corresponding Score: ' + str(img_left_wrist_score))

            print('Right wrist posture')
            right_wrist_status, img_right_wrist_angle = calc_wrist_posture_angle(right_elbow_keypoint_coordinates,
                                                             right_wrist_keypoint_coordinates,
                                                             right_knuckle_keypoint_coordinates)
            print('Right wrist angle: ' + str(img_right_wrist_angle))
            img_right_wrist_score = calc_wrist_score(img_right_wrist_angle)
            print('Right wrist Corresponding Score: ' + str(img_right_wrist_score))

            is_left_wrist_twist = False
            if is_left_wrist_twist:
                img_left_wrist_twist_score = 2
            else:
                img_left_wrist_twist_score = 1

            print('Left Wrist Twist Score: ' + str(img_left_wrist_twist_score))

            is_right_wrist_twist = False
            if is_right_wrist_twist:
                img_right_wrist_twist_score = 2
            else:
                img_right_wrist_twist_score = 1

            print('Right Wrist Twist Score: ' + str(img_right_wrist_twist_score))

            # Neck Posture

            img_neck_status, img_neck_angle = calc_neck_posture_angle(mid_trunk_keypoint_coordinates,
                                                                      neck_keypoint_coordinates,
                                                                      head_keypoint_coordinates)

            print('Neck angle: ' + str(img_neck_angle))
            print('Neck Status: ' + str(img_neck_status))
            img_neck_score = calc_neck_score(img_neck_angle, img_neck_status)
            print('Neck Corresponding Score: ' + str(img_neck_score))

            # Trunk Posture
            img_trunk_angle = calc_trunk_posture_angle(mid_vertical_keypoint_coordinates,
                                                       mid_trunk_keypoint_coordinates,
                                                       neck_keypoint_coordinates)
            print('trunk angle: ' + str(img_trunk_angle))
            img_trunk_score = calc_trunk_score(img_trunk_angle)
            print('Trunk Corresponding Score: ' + str(img_trunk_score))

            # Side body calculation ends
            break
    # Front body starts
    extracted_test_bodies_front = extract_person_from_image(test_img_front)

    if len(extracted_test_bodies_front) > 0:
        print('No of people detected: ' + str(len(extracted_test_bodies_front)))
        for extracted_test_body_front in extracted_test_bodies_front:
            test_label_data_list_front = []
            test_label_data_list_front = decode_pose_estimate_net(test_img_front, extracted_test_body_front,
                                                                  posture_estimation_model)
            # continue
            body_keypoint_df = pd.DataFrame(test_label_data_list_front)
            body_keypoint_df = body_keypoint_df.set_index([2])
            left_shoulder_keypoint_df = body_keypoint_df.loc['left_shoulder']
            left_shoulder_keypoint_coordinates = [left_shoulder_keypoint_df[0], left_shoulder_keypoint_df[1]]
            right_shoulder_keypoint_df = body_keypoint_df.loc['right_shoulder']
            right_shoulder_keypoint_coordinates = [right_shoulder_keypoint_df[0], right_shoulder_keypoint_df[1]]

            mid_shoulder_keypoint_coordinates = [
                abs(left_shoulder_keypoint_df[0] + right_shoulder_keypoint_df[0]) * 0.5,
                abs(left_shoulder_keypoint_df[1] + right_shoulder_keypoint_df[1]) * 0.5]

            neck_keypoint_df = body_keypoint_df.loc['neck']
            neck_keypoint_coordinates = [neck_keypoint_df[0], neck_keypoint_df[1]]
            nose_keypoint_df = body_keypoint_df.loc['nose']
            nose_keypoint_coordinates = [nose_keypoint_df[0], nose_keypoint_df[1]]
            right_eye_keypoint_df = body_keypoint_df.loc['right_eye']
            right_eye_keypoint_coordinates = [right_eye_keypoint_df[0], right_eye_keypoint_df[1]]
            left_eye_keypoint_df = body_keypoint_df.loc['left_eye']
            left_eye_keypoint_coordinates = [left_eye_keypoint_df[0], left_eye_keypoint_df[1]]
            head_keypoint_df = body_keypoint_df.loc['head']
            head_keypoint_coordinates = [head_keypoint_df[0], head_keypoint_df[1]]
            left_elbow_keypoint_df = body_keypoint_df.loc['left_elbow']
            left_elbow_keypoint_coordinates = [left_elbow_keypoint_df[0], left_elbow_keypoint_df[1]]
            right_elbow_keypoint_df = body_keypoint_df.loc['right_elbow']
            right_elbow_keypoint_coordinates = [right_elbow_keypoint_df[0], right_elbow_keypoint_df[1]]
            left_wrist_keypoint_df = body_keypoint_df.loc['left_wrist']
            left_wrist_keypoint_coordinates = [left_wrist_keypoint_df[0], left_wrist_keypoint_df[1]]
            right_wrist_keypoint_df = body_keypoint_df.loc['right_wrist']
            right_wrist_keypoint_coordinates = [right_wrist_keypoint_df[0], right_wrist_keypoint_df[1]]
            left_knuckle_keypoint_df = body_keypoint_df.loc['left_knuckle']
            left_knuckle_keypoint_coordinates = [left_knuckle_keypoint_df[0], left_knuckle_keypoint_df[1]]
            right_knuckle_keypoint_df = body_keypoint_df.loc['right_knuckle']
            right_knuckle_keypoint_coordinates = [right_knuckle_keypoint_df[0], right_knuckle_keypoint_df[1]]

            left_trunk_keypoint_df = body_keypoint_df.loc['left_trunk']
            left_trunk_keypoint_coordinates = [left_trunk_keypoint_df[0], left_trunk_keypoint_df[1]]
            right_trunk_keypoint_df = body_keypoint_df.loc['right_trunk']
            right_trunk_keypoint_coordinates = [right_trunk_keypoint_df[0], right_trunk_keypoint_df[1]]

            mid_trunk_keypoint_coordinates = [abs(left_trunk_keypoint_df[0] + right_trunk_keypoint_df[0]) * 0.5,
                                              abs(left_trunk_keypoint_df[1] + right_trunk_keypoint_df[1]) * 0.5]

            left_knee_keypoint_df = body_keypoint_df.loc['left_knee']
            left_knee_keypoint_coordinates = [left_knee_keypoint_df[0], left_knee_keypoint_df[1]]
            right_knee_keypoint_df = body_keypoint_df.loc['right_knee']
            right_knee_keypoint_coordinates = [right_knee_keypoint_df[0], right_knee_keypoint_df[1]]

            mid_vertical_keypoint_coordinates = [abs(left_trunk_keypoint_df[0] + right_trunk_keypoint_df[0]) * 0.5,
                                                 mid_trunk_keypoint_coordinates[1]*0.5]

            left_ankle_keypoint_df = body_keypoint_df.loc['left_ankle']
            left_ankle_keypoint_coordinates = [left_ankle_keypoint_df[0], left_ankle_keypoint_df[1]]
            right_ankle_keypoint_df = body_keypoint_df.loc['right_ankle']
            right_ankle_keypoint_coordinates = [right_ankle_keypoint_df[0], right_ankle_keypoint_df[1]]

            _is_shoulder_raised = user_param[2]

            print('Is Shoulder Raised: ' + str(_is_shoulder_raised))
            if _is_shoulder_raised:
                img_left_upper_arm_score = img_left_upper_arm_score + 1
                print('left upper arm score updated due to raise: ' + str(img_left_upper_arm_score))
                img_right_upper_arm_score = img_right_upper_arm_score + 1
                print('right upper arm score updated due to raise: ' + str(img_right_upper_arm_score))

            img_left_upper_arm_abduction_angle = calc_upper_arm_abduction(left_trunk_keypoint_coordinates,
                                                                          left_shoulder_keypoint_coordinates,
                                                                          left_elbow_keypoint_coordinates)
            # Shoulder is abducted if upper arm abduction>45*
            _is_left_shoulder_abducted = True if abs(img_left_upper_arm_abduction_angle) >= 45 else False
            print('Is left Shoulder abducted: ' + str(_is_left_shoulder_abducted))
            if _is_left_shoulder_abducted:
                img_left_upper_arm_score = img_left_upper_arm_score + 1
                print('left upper arm score updated due to shoulder abduction: ' + str(img_left_upper_arm_score))

            img_right_upper_arm_abduction_angle = calc_upper_arm_abduction(right_trunk_keypoint_coordinates,
                                                                           right_shoulder_keypoint_coordinates,
                                                                           right_elbow_keypoint_coordinates)
            _is_right_shoulder_abducted = True if abs(img_right_upper_arm_abduction_angle) >= 45 else False
            print('Is right Shoulder abducted: ' + str(_is_right_shoulder_abducted))
            if _is_right_shoulder_abducted:
                img_right_upper_arm_score = img_right_upper_arm_score + 1
                print('right upper arm score updated due to shoulder abduction: ' + str(img_right_upper_arm_score))

            if img_right_upper_arm_score == 0:
                print('Right Upper arm score was 1 and made 0 because posture is leaned and rested')
                img_right_upper_arm_score = 1
            if img_left_upper_arm_score == 0:
                print('Left Upper arm score was 1 and made 0 because posture is leaned and rested')
                img_left_upper_arm_score = 1

            print('Final left upper arm score: ' + str(img_left_upper_arm_score))
            print('Final right upper arm score: ' + str(img_right_upper_arm_score))

            img_lower_arm_working_midline = calc_lower_arm_work_midline(right_shoulder_keypoint_coordinates,
                                                                        left_shoulder_keypoint_coordinates,
                                                                        right_wrist_keypoint_coordinates,
                                                                        left_wrist_keypoint_coordinates)
            print('is lower arm working midline ' + str(img_lower_arm_working_midline))
            if img_lower_arm_working_midline:
                print('Lower arm score updated due to arm working midline posture')
                img_left_lower_arm_score = img_left_lower_arm_score + 1
                img_right_lower_arm_score = img_right_lower_arm_score + 1

            print('Final left lower arm score: ' + str(img_left_lower_arm_score))
            print('Final right lower arm score: ' + str(img_right_lower_arm_score))

            # is_left_wrist_bent_away_midline = calc_wrist_bent(left_elbow_keypoint_coordinates, left_wrist_keypoint_coordinates,
            #                                                   left_knuckle_keypoint_coordinates,'front')
            is_left_wrist_bent_away_midline = False

            print('Is left wrist bent away from mid: ' + str(is_left_wrist_bent_away_midline))
            if is_left_wrist_bent_away_midline:
                img_left_wrist_score = img_left_wrist_score + 1
                print('Left Wrist Score updated because wrist bent away midline: ' + str(img_left_wrist_score))
                img_right_wrist_score = img_right_wrist_score + 1
                print('Right Wrist Score updated because wrist bent away midline: ' + str(img_right_wrist_score))

            if nose_keypoint_coordinates[0] > 0:
                img_neck_twist = calc_neck_twist(left_shoulder_keypoint_coordinates,
                                                 right_shoulder_keypoint_coordinates,
                                                 nose_keypoint_coordinates)
                img_neck_side_bending = calc_neck_side_bending(mid_trunk_keypoint_coordinates,
                                                               neck_keypoint_coordinates,
                                                               head_keypoint_coordinates)
            else:
                print('Nose coordinates are unavailable probably because face is undetected')
                img_neck_twist = False
                img_neck_side_bending = False

            print('is neck twisted: ' + str(img_neck_twist))
            if img_neck_twist:
                img_neck_score = img_neck_score + 1
                print('Neck Score updated due to neck twisting: ' + str(img_neck_score))

            print('is neck side bending: ' + str(img_neck_side_bending))
            if img_neck_side_bending:
                img_neck_score = img_neck_score + 1
                print('Neck Score updated due to side bending: ' + str(img_neck_score))

            img_trunk_side_bending = calc_trunk_posture_angle(mid_vertical_keypoint_coordinates,
                                                       mid_trunk_keypoint_coordinates,
                                                       neck_keypoint_coordinates)

            is_img_trunk_side_bending = False
            if img_trunk_side_bending == 180:
                is_img_trunk_side_bending = False
            elif abs(img_trunk_side_bending)>15:
                is_img_trunk_side_bending = True

            print('is trunk side bent: ' + str(is_img_trunk_side_bending))
            if is_img_trunk_side_bending:
                img_trunk_score = img_trunk_score + 1
                print('Trunk Score updated due to side bending: ' + str(img_trunk_score))

            img_trunk_twist = calc_trunk_twist(left_shoulder_keypoint_coordinates, right_shoulder_keypoint_coordinates,
                                               left_trunk_keypoint_coordinates, right_trunk_keypoint_coordinates,
                                               img_trunk_side_bending)
            print('is trunk twist: ' + str(img_trunk_twist))
            if img_trunk_twist:
                img_trunk_score = img_trunk_score + 1
                print('Trunk Score updated due to twisting: ' + str(img_trunk_score))

            img_leg_support_status = calc_leg_support(left_ankle_keypoint_coordinates, right_ankle_keypoint_coordinates,
                                                      left_trunk_keypoint_coordinates, right_trunk_keypoint_coordinates)
            print('is leg supported: ' + str(img_leg_support_status))
            if img_leg_support_status:
                img_leg_posture_score = 1
            else:
                img_leg_posture_score = 2
            print('Corresponding Score: ' + str(img_leg_posture_score))

            # front view calculation ends
            break

        # Main Rula score estimation starts
        print('Finding posture score for left side group A')
        print(img_left_upper_arm_score, img_left_lower_arm_score, img_left_wrist_score, img_left_wrist_twist_score)

        left_posture_score_a = find_rula_posture_score_a(upper_arm_score=img_left_upper_arm_score,
                                                         lower_arm_score=img_left_lower_arm_score,
                                                         wrist_posture_score=img_left_wrist_score,
                                                         wrist_twist_score=img_left_wrist_twist_score)

        print('Finding posture score for right side group A')
        print(img_right_upper_arm_score, img_left_lower_arm_score, img_right_wrist_score, img_right_wrist_twist_score)

        right_posture_score_a = find_rula_posture_score_a(upper_arm_score=img_right_upper_arm_score,
                                                          lower_arm_score=img_right_lower_arm_score,
                                                          wrist_posture_score=img_right_wrist_score,
                                                          wrist_twist_score=img_right_wrist_twist_score)

        print('Finding posture score for group B')
        print(img_neck_score, img_trunk_score, img_leg_posture_score)

        posture_score_b = find_rula_posture_score_b(neck_posture_score=img_neck_score,
                                                    trunk_posture_score=img_trunk_score,
                                                    leg_posture_score=img_leg_posture_score)

        print('Finding grand score for left side')
        print(left_posture_score_a, posture_score_b)

        left_grand_posture_score = find_rula_grand_score(upper_body_posture_score=left_posture_score_a,
                                                         lower_body_posture_score=posture_score_b)

        print('Finding grand score for right side')
        print(right_posture_score_a, posture_score_b)

        right_grand_posture_score = find_rula_grand_score(upper_body_posture_score=right_posture_score_a,
                                                          lower_body_posture_score=posture_score_b)

        rula_final_grand_score = max(left_grand_posture_score, right_grand_posture_score)

        print('Rula Grand Score ' + str(rula_final_grand_score))

    print('Completed')
    rula_result_dict['left_upper_arm_score'] = int(img_left_upper_arm_score)
    rula_result_dict['right_upper_arm_score'] = int(img_right_upper_arm_score)
    rula_result_dict['left_lower_arm_score'] = int(img_left_lower_arm_score)
    rula_result_dict['right_lower_arm_score'] = int(img_right_lower_arm_score)
    rula_result_dict['left_wrist_score'] = int(img_left_wrist_score)
    rula_result_dict['right_wrist_score'] = int(img_right_wrist_score)
    rula_result_dict['left_wrist_twist_score'] = int(img_left_wrist_twist_score)
    rula_result_dict['right_wrist_twist_score'] = int(img_right_wrist_twist_score)
    rula_result_dict['trunk_score'] = int(img_trunk_score)
    rula_result_dict['neck_score'] = int(img_neck_score)
    rula_result_dict['leg_score'] = int(img_leg_posture_score)
    rula_result_dict['right_posture_score_a'] = int(right_posture_score_a)
    rula_result_dict['left_posture_score_a'] = int(left_posture_score_a)
    rula_result_dict['right_posture_score_b'] = int(posture_score_b)
    rula_result_dict['left_grand_score'] = int(left_grand_posture_score)
    rula_result_dict['right_grand_score'] = int(right_grand_posture_score)
    return rula_result_dict


model_class = read_classes('static/resource/model_class_annotation.txt')

model_class_id_list = {k: v for k, v in zip(model_class, list(range(1, len(model_class)+1)))}
print(model_class_id_list)

# define the yolo v3 model
yolov3 = make_yolov3_model()
# load the weights
weight_reader = WeightReader('static/weight_files/person_detector_model.weights')
# set the weights
weight_reader.load_weights(yolov3)

# save the model to file
print('Person detector model loaded')
posture_estimation_model = posture_estimation_model_func()
posture_estimation_model.load_weights('static/weight_files/full_body_posture_aug_model_weights.hdf5')
print('Pose detector model loaded')