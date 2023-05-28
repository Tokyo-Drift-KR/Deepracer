import math

# import csv
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# 
# csv file로 만들거나 pyplot을 통해 border, waypoint를 그려줌
# def make_csv_points(waypoints, left_border, right_border):
#     temp = []
#     for i in range(len(waypoints)):
#         temp.append(waypoints[i] + left_border[i] + right_border[i])
#         
#     waypoints_x, waypoints_y = [], []
#     borders_x, borders_y = [], []
#     
#     for i in range(len(waypoints)):
#         (x, y), (lx, ly), (rx, ry) = waypoints[i], left_border[i], right_border[i]
#         waypoints_x.append(x)
#         waypoints_y.append(y)
#         borders_x.append(lx)
#         borders_x.append(rx)
#         borders_y.append(ly)
#         borders_y.append(ry)
#     
#     plt.scatter(waypoints_x, waypoints_y, color="blue", s = 0.1)
#     plt.scatter(borders_x, borders_y, color="red", s = 0.1)
#     plt.show()
#     
#     header = ["x", "y", "lx", "ly", "rx", "ry"]
#     name = "points.csv"
#     with open(name, "w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerows(header)
#         writer.writerows(temp)
#     return

def CCW(x1, y1, x2, y2, x3, y3):
    # a (x1, y1) | b (x2, y2)
    # c (x3, y3)
    # ab 를 기준으로 c의 방향성을 찾음.
    # need three coordinate in argument.
    ret = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    if ret == 0:
        # 평행한 경우
        return 0
    elif ret < 0:
        # 시계 방향
        return -1
    elif ret > 0:
        # 반시계 방향
        return 1


def isIntersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # a (x1, y1) | b (x2, y2)
    # c (x3, y3) | d (x4, y4)
    ab = CCW(x1, y1, x2, y2, x3, y3) * CCW(x1, y1, x2, y2, x4, y4)
    cd = CCW(x3, y3, x4, y4, x1, y1) * CCW(x3, y3, x4, y4, x2, y2)

    if (ab == 0 and cd == 0):
        if x1 + y1 > x2 + y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        if x3 + y3 > x4 + y4:
            x3, y3, x4, y4 = x4, y4, x3, y3

        return x1 + y1 <= x4 + y4 and x3 + y3 <= x2 + y2

    return ab <= 0 and cd <= 0


def dist(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def cal_degree(x, y):
    degree = math.degrees(math.atan2(y, x))
    return degree


def get_waypoints_ordered_in_driving_direction(params):
    # waypoints are always provided in counter clock wise order
    if params['is_reversed']:  # driving clock wise.
        return list(reversed(params['waypoints']))
    else:  # driving counter clock wise.
        return params['waypoints']


def mod_360(angle):
    n = math.floor(angle / 360.0)
    ret = angle - n * 360.0

    if ret > 180:
        return ret - 360
    return ret

def border_waypoints(now_coor, next_coor, half_width):
    """use orthogonal vector, and linear calculation"""
    x, y = now_coor
    nx, ny = next_coor
    hw = half_width

    vector = (nx - x, ny - y)
    ortho = (vector[1], -vector[0])

    left = (x + hw * ortho[0], y + hw * ortho[1])
    right = (x - hw * ortho[0], y - hw * ortho[1])

    return left, right


def up_sample(waypoints, total_cnt):
    """ with linear interpolation
        add_point : how many new data between prev_dot, now_dot"""
    ret = []

    pivot = len(waypoints)
    step = (pivot - 1) / (total_cnt - 1)
    for i in range(total_cnt):
        idx = int(i * step)

        if idx == pivot - 1:
            ret.append(waypoints[-1])
        else:
            ratio = i * step - idx
            x = waypoints[idx][0] + ratio * (waypoints[idx + 1][0] - waypoints[idx][0])
            y = waypoints[idx][1] + ratio * (waypoints[idx + 1][1] - waypoints[idx][1])
            ret.append((x, y))

    return ret


def down_sample(waypoints, ratio):
    """ratio : it will makes the cnt in waypoint as len(waypoint) -> len(waypoint) / ratio"""
    ret = []
    for i in range(0, len(waypoints), ratio):
        ret.append(waypoints[i])
    return ret

def check_maximal_waypoint(params):
    ret_reward = 0.01

    waypoint = get_waypoints_ordered_in_driving_direction(params)

    # to generalize upsample (make 5000 waypoint)
    # -> downsample (make 1000 waypoint)
    # now all the track composed with 1000 waypoint.
    # car will select the action with at least 10 waypoint.
    waypoint = up_sample(waypoint, 5000)
    waypoint = down_sample(waypoint, 5)

    # border_waypoints((0, 0), (2, 0), 2)
    # border_waypoints((0, 0), (1, -1), 2)
    # border_waypoints((0, 0), (-1, 1), 2)
    # border_waypoints((0, 0), (-1, -1), 2)
    # making border line
    h_width = params["track_width"] / 2
    cnt = len(waypoint)
    left_border, right_border = [], []
    for idx in range(1, cnt):
        prev, now = waypoint[idx - 1], waypoint[idx % cnt]
        l_d, r_d = border_waypoints(prev, now, h_width)
        left_border.append(l_d)
        right_border.append(r_d)

    # make waypoint, borders in a closest one.
    dist_waypoint = []
    for idx in range(1, cnt):
        prev, now = waypoint[idx - 1], waypoint[idx]
        dist_waypoint.append(dist(prev, now))

    i_closest = dist_waypoint.index(min(dist_waypoint))
    waypoint = waypoint[i_closest:] + waypoint[:i_closest]
    left_border = left_border[i_closest:] + left_border[:i_closest]
    right_border = right_border[i_closest:] + right_border[:i_closest]

    show_cnt = 500
    # make_csv_points(waypoint[:show_cnt], left_border[:show_cnt], right_border[:show_cnt])
    # make the line between car - waypoint.
    # check the border does intersect with this.

    # 1. check the line intersect with border -> if does, it has right or left turn.
    # 2. but the intersect happens far away (it checks with variable "treat_cnt") -> keep center?

    # goal is find the fartest waypoint. (that doesnt have intersect one)

    # checking the whole waypoint is useless.
    # need the threshold to treat it as a dataset.
    treat_cnt = 80
    x, y = params["x"], params["y"]
    for i in range(1, treat_cnt + 1):
        nx, ny = waypoint[i]
        for j in range(1, treat_cnt + 1):
            (l1_nx, l1_ny), (l2_nx, l2_ny) = left_border[j - 1], left_border[j]
            (r1_nx, r1_ny), (r2_nx, r2_ny) = right_border[j - 1], right_border[j]

            l_inter = isIntersect(x, y, nx, ny, l1_nx, l1_ny, l2_nx, l2_ny)
            r_inter = isIntersect(x, y, nx, ny, r1_nx, r1_ny, r2_nx, r2_ny)
            if l_inter or r_inter:
                return nx, ny

    return waypoint[treat_cnt]


def check_steer_angle(params):
    nx, ny = check_maximal_waypoint(params)
    x, y = params["x"], params["y"]
    heading = params["heading"]

    target_angle = cal_degree(nx - x, ny - y)
    steer = target_angle - heading
    return mod_360(steer)

def ideal_speed(angle):
    min_angle = 0
    max_angle = 15
    min_speed = 1.2
    max_speed = 3

    mapped_angle = ((angle - min_angle) / (max_angle - min_angle)) * (max_speed - min_speed)
    speed = (mapped_angle / max_angle) * (max_speed - min_speed) + min_speed

    return speed

def reward_degree(params):
    ideal_angle = check_steer_angle(params)
    real_angle = params["steering_angle"]
    err = (real_angle - ideal_angle) / 50
    ret = 1.0 - abs(err)
    return ideal_angle, max(ret, 0.01)

def reward_speed(ideal_speed, params):
    real_speed = params["speed"]
    err = (real_speed - ideal_speed) / 3
    ret = 1.0 - abs(err)
    return max(ret, 0.01)

def reward_function(params):
    speed, ret_reward = reward_degree(params)
    # ret_reward += reward_speed(speed, params)
    return float(ret_reward)


def get_test_params():
    return {
        'x': 0.75,
        'y': 0.02,
        'heading': 160.0,
        "distance_from_center": 0.1,
        'track_width': 0.45,
        "speed": 3.0,
        'is_reversed': False,
        'steering_angle': 0.0,
        "closest_waypoints": [1, 2],
        'waypoints': [
            [0.75, -0.7],
            [1.0, 0.0],
            [0.7, 0.52],
            [0.58, 0.7],
            [0.48, 0.8],
            [0.15, 0.95],
            [-0.1, 1.0],
            [-0.7, 0.75],
            [-0.9, 0.25],
            [-0.9, -0.55],
        ]
    }


def test_reward():
    params = get_test_params()

    reward = reward_function(params)

    print("test_reward: {}".format(reward))

    assert reward > 0.0


def test_get_target_point():
    # result = get_target_point(get_test_params(), 0.45 * 0.9)
    expected = [0.33, 0.86]
    eps = 0.1

    # print("get_target_point: x={}, y={}".format(result[0], result[1]))

    # assert dist(result, expected) < eps


def test_get_target_steering():
    result = reward_degree(get_test_params())
    expected = 46
    eps = 1.0

    print("get_target_steering={}".format(result))

    # assert abs(result - expected) < eps


def test_angle_mod_360():
    eps = 0.001

    # assert abs(-90 - angle_mod_360(270.0)) < eps
    # assert abs(-179 - angle_mod_360(181)) < eps
    # assert abs(0.01 - angle_mod_360(360.01)) < eps
    # assert abs(5 - angle_mod_360(365.0)) < eps
    # assert abs(-2 - angle_mod_360(-722)) < eps


def test_upsample():
    params = get_test_params()
    print(repr(up_sample(params['waypoints'], 2)))


def test_score_steer_to_point_ahead():
    params_l_45 = {**get_test_params(), 'steering_angle': +45}
    params_l_15 = {**get_test_params(), 'steering_angle': +15}
    params_0 = {**get_test_params(), 'steering_angle': 0.0}
    params_r_15 = {**get_test_params(), 'steering_angle': -15}
    params_r_45 = {**get_test_params(), 'steering_angle': -45}

    # sc = score_steer_to_point_ahead

    # # 0.828, 0.328, 0.078, 0.01, 0.01
    # print("Scores: {}, {}, {}, {}, {}".format(sc(params_l_45), sc(params_l_15), sc(params_0), sc(params_r_15),
    #                                           sc(params_r_45)))


def run_tests():
    test_angle_mod_360()
    test_reward()
    test_upsample()
    test_get_target_point()
    test_get_target_steering()
    test_score_steer_to_point_ahead()

    print("All tests successful")

#run_tests()