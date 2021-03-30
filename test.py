import numpy as np
import math

def calc_upper_arm_angle(point_a, base_point, point_b):
    trunk_joint_coordinate = np.array([point_a[0], point_a[1]])
    shoulder_joint_coordinate = np.array([base_point[0], base_point[1]])
    elbow_joint_coordinate = np.array([point_b[0], point_b[1]])

    ba = trunk_joint_coordinate - shoulder_joint_coordinate
    bc = elbow_joint_coordinate - shoulder_joint_coordinate

    # positive if elbow is behind shoulder (Anti-clockwise) i.e extension
    # negative: flexion
    upper_arm_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print(upper_arm_angle)
    return upper_arm_angle



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

    wrist_angle = abs(wrist_angle)

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
    neck_angle = int(np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc))))
    print('Func neck angle: ' + str(neck_angle))
    if neck_angle==180:
        neck_status = True
        print('Neck neutral')
        neck_angle = 180 - neck_angle
    elif 180 > neck_angle > 0:
        neck_status = True
        print('Neck flexion')
    elif -180 < neck_angle < 0:
        neck_status = False
        print('Neck Extension')
        neck_angle = abs(neck_angle)
    print('Neck angle: ' + str(neck_angle))
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

def calc_upper_arm_abduction(point_a, base_point, point_b):
    trunk_joint_coordinate = np.array([point_a[0], point_a[1]])
    shoulder_joint_coordinate = np.array([base_point[0], base_point[1]])
    elbow_joint_coordinate = np.array([point_b[0], point_b[1]])

    ba = trunk_joint_coordinate - shoulder_joint_coordinate
    bc = elbow_joint_coordinate - shoulder_joint_coordinate

    # positive if elbow is behind shoulder (Anti-clockwise) i.e extension
    # negative: flexion
    upper_arm_angle = np.degrees(np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc)))
    print(upper_arm_angle)
    return upper_arm_angle

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


# distance between head and both shoulders must be same
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

    if abs(dist_1 - dist_2) > 15:
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

#left body
# trunk=[69,166]
# neck=[50,97]
# head=[64,80]
#right body
trunk=[69,166]
neck=[89,97]
head=[99,80]

calc_neck_posture_angle(trunk, neck, head)
