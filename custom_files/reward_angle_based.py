import numpy as np
import math

eps = 1e-4

########################################################
################### UTILITY FUNTIONS ###################
########################################################

# rotate array to set start_index to 0
def rotate_array(arr, start_index):
    start_index = start_index % len(arr)
    return np.array(arr[start_index:] + arr[:start_index])


# calculate border_waypoints with waypoint and road width
def border_waypoints(waypoints, half_width):
    inner_border_points = []
    outer_border_points = []

    for i in range(len(waypoints)):
        prev_point = waypoints[i - 1] if i > 0 else waypoints[-1]
        current_point = waypoints[i]
        next_point = waypoints[(i + 1) % len(waypoints)]

        # 현재 점에서의 방향 벡터 계산
        direction_vector = np.array(next_point) - np.array(current_point)
        direction_vector /= np.linalg.norm(direction_vector)

        # 외부 벡터 계산
        outer_normal = np.array([-direction_vector[1], direction_vector[0]])

        # 내부 및 외부 경계 계산
        inner_border_point = current_point - half_width / 2 * outer_normal
        outer_border_point = current_point + half_width / 2 * outer_normal

        inner_border_points.append(inner_border_point.tolist())
        outer_border_points.append(outer_border_point.tolist())

    return inner_border_points, outer_border_points

# check validation if the line goes out of border or not
def is_steering_valid(x1, y1, x2, y2, inner_border_points, outer_border_points):
    # calculate function's slope and intercept
    # y = ax + b
    a = (y2 - y1) / (x2 - x1 + eps)
    b = y1 - a * x1

    # check validation left / right set
    inner_validation = np.array(a * np.array(inner_border_points)[:, 0] + b) > np.array(inner_border_points)[:, 1]
    outer_validation = np.array(a * np.array(outer_border_points)[:, 0] + b) > np.array(outer_border_points)[:, 1]

    try:
        if np.all(inner_validation == inner_validation[0]) and np.all(outer_validation == outer_validation[0]):
            if any(inner_validation & outer_validation):
                return False
            else:
                return True
        else:
            return False
    except:
        return False


########################################################
################### CALCULATE REWARD ###################
########################################################

# get optimal steering angle
def get_target_steering(params, max_idx):
    # get parameters
    x, y, waypoints = params['x'], params['y'], params['waypoints']
    inner_points, outer_points = border_waypoints(waypoints, params["track_width"] / 2)

    # rotate arrays
    rotated_waypoints = rotate_array(waypoints, params['closest_waypoints'][1])
    rotated_outer_points = rotate_array(outer_points, params['closest_waypoints'][1])
    rotated_inner_points = rotate_array(inner_points, params['closest_waypoints'][1])

    # Find the optimal index which is in rotated array with Binary Search
    s = 0
    e = max_idx
    target_idx = 0

    while s <= e:
        m = (s + e) // 2

        if not is_steering_valid(x, y, rotated_waypoints[m][0], rotated_waypoints[m][1], rotated_outer_points[:m], rotated_inner_points[:m]):
            e = m - 1
        else:
            s = m + 1
            target_idx = max(target_idx, m)

    # calculate angle between current (x, y) and target idx
    x1, y1 = rotated_waypoints[0][0], rotated_waypoints[0][1]
    x2, y2 = rotated_waypoints[target_idx - 1][0], rotated_waypoints[target_idx - 1][1]

    return np.rad2deg(math.atan2((y2 - y1), (x2 - x1)))

def reward_function(params):
    # calculate optimal steering angle of Deepracer
    target_steering = get_target_steering(params, 20)
    steering_err = abs(params['heading'] - target_steering)
  
    error = steering_err / 60.0  
  
    score = 1.0 - abs(error)
    return max(score, 0.01)
