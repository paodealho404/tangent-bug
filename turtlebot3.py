import math
from controller import Supervisor, Robot
from enum import Enum
import numpy as np
import time


class State(Enum):
    MOTION_TO_GOAL = "In motion!"
    BOUNDARY_FOLLOWING = "Following the boundary!"
    REACHED = "Reached the goal!"
    STUCK = "Impossible to reach the goal!"
    MOVING = "Moving to the point!"


class RobotMotionController:
    def __init__(self, robot, super):

        self._robot = robot
        self._super = super
        self._leftMotor = self._super.getDevice("left wheel motor")
        self._rightMotor = self._super.getDevice("right wheel motor")
        self._timestep = self._super.getBasicTimeStep()
        self._kp = 3 * np.pi

        self._L = 0.16
        self._R = 0.033
        self._v = 0.2
        self._MAX_MOTOR_SPEED = 6.67

        self._theta = self._array_to_euler_angles(self._robot.getOrientation())[2]
        self._pos_x = self._robot.getPosition()[0]
        self._pos_y = self._robot.getPosition()[1]

        self._dist_error = self._L / 2
        self._prev_motion_vec = np.array([1e-6, 1e-6])
        self._prev_motion_ang = 0

    def _array_to_euler_angles(self, arr, in_degrees=False):
        sy = math.sqrt(arr[0] ** 2 + arr[3] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(arr[6], arr[7])
            y = math.atan2(-arr[8], sy)
            z = math.atan2(arr[3], arr[0])
        else:
            x = math.atan2(-arr[5], arr[4])
            y = math.atan2(-arr[8], sy)
            z = 0

        angles = [x, y, z]

        if in_degrees:
            angles = [math.degrees(angle) for angle in angles]

        return angles

    def _get_clamp_speed(self, x):
        if abs(x) >= self._MAX_MOTOR_SPEED:
            return self._MAX_MOTOR_SPEED * (x / abs(x))
        return x

    def _get_both_velocities_from_angular_change(self, ang):
        v_l = ((2 * self._v) - (ang * self._L)) / (2 * self._R)
        v_r = ((2 * self._v) + (ang * self._L)) / (2 * self._R)
        return self._get_clamp_speed(v_l), self._get_clamp_speed(v_r)

    def get_distance_to_target(self, x, y):
        dist = ((x - self._pos_x) ** 2 + (y - self._pos_y) ** 2) ** 0.5

        if dist > self._dist_error:
            return dist
        return 0

    def _get_angle_to_target_in_radians(self, x, y):
        ang = math.atan2(y - self._pos_y, x - self._pos_x)
        return ang

    def update_current_xy(self):
        self._pos_x = self._robot.getPosition()[0]
        self._pos_y = self._robot.getPosition()[1]

    def update_current_theta(self):
        self._theta = self._array_to_euler_angles(self._robot.getOrientation(), False)[
            2
        ]

    def go_to(self, x, y):

        ang = self._get_angle_to_target_in_radians(x, y)
        erro = ang - self._theta
        out = self._kp * erro

        v_l, v_r = self._get_both_velocities_from_angular_change(out)

        self._leftMotor.setVelocity(v_l)
        self._rightMotor.setVelocity(v_r)

    def stop(self):
        self._leftMotor.setPosition(float("inf"))
        self._rightMotor.setPosition(float("inf"))
        self._leftMotor.setVelocity(0.0)
        self._rightMotor.setVelocity(0.0)


class RobotTrajectoryPlanner:
    def __init__(self, robot, target_x, target_y):
        self._robot = robot
        self._target_x = target_x
        self._target_y = target_y
        self._d_reach = 0
        self._d_followed = 0
        self._bound_pos = []
        self._lidar_distances = []
        self._last_motion_ang = 0
        self._last_motion_vec = np.array([1e-6, 1e-6])
        self._L = 0.16
        self._err = 0.1
        self._check_again = False

    def set_lidar_info(self, max_range, offset):
        self._lidar_max_range = max_range

    def _array_to_euler_angles(self, arr, in_degrees=False):
        sy = math.sqrt(arr[0] ** 2 + arr[3] ** 2)

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(arr[6], arr[7])
            y = math.atan2(-arr[8], sy)
            z = math.atan2(arr[3], arr[0])
        else:
            x = math.atan2(-arr[5], arr[4])
            y = math.atan2(-arr[8], sy)
            z = 0

        angles = [x, y, z]

        if in_degrees:
            angles = [math.degrees(angle) for angle in angles]

        return angles

    def _find_obstacles(self, lidar_reads):
        _valid_angles = np.nonzero(np.array(lidar_reads) < self._lidar_max_range * 0.1)[0]

        _lower_boundaries = []
        _upper_boundaries = []
        _obstacles = []

        for i, curr_angle in enumerate(_valid_angles):
            _next_valid_angle_c_wise = _valid_angles[(i + 1) % len(_valid_angles)]
            _next_angle_c_wise = curr_angle + 1

            _next_valid_angle_cc_wise = _valid_angles[(i - 1) % len(_valid_angles)]
            _next_angle_cc_wise = curr_angle - 1

            # If next_angle clockwise is not the next lidar read clockwise angle, then we have a lower boundary
            if _next_angle_c_wise != _next_valid_angle_c_wise:
                _lower_boundaries.append((curr_angle) % 360)

            # If next_angle counter clockwise is not the next lidar read counter clockwise angle, then we have an upper boundary
            if _next_angle_cc_wise != _next_valid_angle_cc_wise:
                _upper_boundaries.append((curr_angle) % 360)

        for lim in zip(_upper_boundaries, _lower_boundaries):
            if lim[0] != lim[1]:
                _obstacles.append((lim[0], lim[1]))

        if not _obstacles:
            return _obstacles

        if abs(_obstacles[-1][1] - _obstacles[0][0]) == 1:
            _obstacles[0] = (_obstacles[0][1], _obstacles[-1][0])
            _obstacles.pop()

        # print(_obstacles)
        return _obstacles

    def _range2cart(self, x):
        _Tor = np.array(
            [
                [
                    [np.cos(self._theta), -np.sin(self._theta), 0, self._pos_x],
                    [np.sin(self._theta), np.cos(self._theta), 0, self._pos_y],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ]
        )

        _rel_robot_dx = self._lidar_distances[x] * np.cos((180 - x) * np.pi / 180)
        _rel_robot_dy = self._lidar_distances[x] * np.sin((180 - x) * np.pi / 180)

        _rel_robot_dx = (
            self._lidar_max_range if np.isinf(_rel_robot_dx) else _rel_robot_dx
        )
        _rel_robot_dy = (
            self._lidar_max_range if np.isinf(_rel_robot_dy) else _rel_robot_dy
        )

        _dist = np.array([[_rel_robot_dx, _rel_robot_dy, 0, 1]]).reshape(4, 1)

        _dist_world = (_Tor @ _dist).ravel()

        return _dist_world[:2]

    def _get_cartesian_obstacle(self, obstacle):
        oi_mat = np.empty((len(obstacle), 2))
        for i, x in enumerate(obstacle):
            oi_mat[i, :] = self._range2cart(int(x))

        return oi_mat[::-1]

    def _deg_to_target(self):
        theta = np.arctan2(self._target_y - self._pos_y, self._target_x - self._pos_x)

        theta = theta % (2 * np.pi)
        return theta

    def _is_angle_between(self, angulo1, angulo2, angulo3, gap_degrees=0):
        # Normalizando os ângulos para garantir que estejam entre 0 e 2pi
        # Normaliza os ângulos para o intervalo [0, 2*pi]
        gap_radians = np.radians(gap_degrees)
        
        # Normalizando os ângulos para garantir que estejam entre 0 e 2*pi
        angulo1 = angulo1 % (2 * np.pi)
        angulo2 = angulo2 % (2 * np.pi)
        angulo3 = angulo3 % (2 * np.pi)

        # Calcula a diferença angular entre os ângulos
        dif_angulo12 = (angulo2 - angulo1) % (2 * np.pi)
        dif_angulo13 = (angulo3 - angulo1) % (2 * np.pi)

        # Adiciona ou subtrai o gap da diferença angular entre angulo1 e angulo2
        dif_angulo12_gap = dif_angulo12 + gap_radians

        # Verifica se o terceiro ângulo está entre os dois primeiros, considerando o gap
        if dif_angulo12 < np.pi:
            return 0 < dif_angulo13 < dif_angulo12_gap
        else:
            return dif_angulo13 > 0 or dif_angulo13 < dif_angulo12_gap  

    def _get_tangent_vector(self, obstacles):
        reg_idx = []

        if isinstance(obstacles, tuple):
            obstacles = [obstacles]

        for lims in obstacles:
            lim_inf, lim_sup = lims
            aux = [lim_inf, lim_sup]
            lim_inf = min(aux)
            lim_sup = max(aux)

            idxs = np.unique(
                np.linspace(
                    lim_inf, lim_sup, -(-1 * (lim_sup - lim_inf) // 4) + 1, dtype=int
                )
            )
            reg_idx = np.r_[np.array(reg_idx), idxs]

        oi_mat = self._get_cartesian_obstacle(reg_idx)
        pos_vec = np.array([self._pos_x, self._pos_y])
        _dist_to_obj = np.linalg.norm(pos_vec - oi_mat, axis=1) ** 2

        min_idx = np.argmin(_dist_to_obj)
        max_idx = np.argmax(_dist_to_obj)

        min_dist = _dist_to_obj[min_idx]

        # print(oi_mat[min_idx])

        h = 0.1
        _sum, _div = np.array([0.0, 0.0]), 0

        for i, a in enumerate(_dist_to_obj):
            _temp = np.exp((min_dist - a) / (2 * h**2))
            _sum += _temp * (pos_vec - oi_mat[i, :])
            _div += _temp

        rot90 = np.array(
            [
                [np.cos(self._last_motion_ang), -np.sin(self._last_motion_ang)],
                [np.sin(self._last_motion_ang), np.cos(self._last_motion_ang)],
            ]
        )
        heuristic = _sum / _div

        tangent = (rot90 @ heuristic.reshape(-1, 1)).ravel()
        closest_point = pos_vec - heuristic

        return tangent, closest_point

    def _check_any_obstacle_ahead(self, obstacles, target_x, target_y):
        _ang_to_check = self._deg_to_target()

        _obj = None
        _blocking = False

        for i, obj in enumerate(obstacles):
            _lim_sup = (180 - obj[0]) * np.pi / 180
            _lim_inf = (180 - obj[1]) * np.pi / 180
            
            if self._is_angle_between(_lim_inf, _lim_sup, self._theta-_ang_to_check, 17):
                _obj = obj
                _obj_coords = self._get_cartesian_obstacle(list(_obj[::-1]))

                _dist_d_1 = np.linalg.norm(
                    _obj_coords[0] - np.array([self._pos_x, self._pos_y])
                )
                _dist_d_2 = np.linalg.norm(
                    _obj_coords[1] - np.array([self._pos_x, self._pos_y])
                )
                _dist_rob_to_target = np.linalg.norm(
                    np.array([self._pos_x, self._pos_y])
                    - np.array([target_x, target_y])
                )

                if _dist_d_1 < _dist_rob_to_target or _dist_d_2 < _dist_rob_to_target:
                    print(
                        "Obstacle ahead!",
                        round(_lim_inf * 180 / np.pi),
                        round(_lim_sup * 180 / np.pi),
                        round(_ang_to_check * 180 / np.pi),
                    )

                    _blocking = True
                    return _blocking, i

        return _blocking, _obj

    def _safety_distance(self, closest_point, tangent):
        _vec_robot_to_obj = closest_point - [self._pos_x, self._pos_y]
        _obj_dist = np.linalg.norm(_vec_robot_to_obj)

        if _obj_dist == 0:
            _obj_dist = 1e-3

        oi_safe = ((_vec_robot_to_obj + tangent * 8)) - self._L
        return oi_safe

    def _choose_object(self, obstacles):
        _goal_vec = np.array([self._target_x, self._target_y])

        _tangent, _closest_point = self._get_tangent_vector(obstacles)
        _obj_to_follow = self._safety_distance(_closest_point, _tangent)

        self._d_reach = np.linalg.norm(_goal_vec - _closest_point)

        if self._d_reach >= self._d_followed:
            self._d_followed = self._d_reach

        return _obj_to_follow

    def update_current_xy(self):
        self._pos_x = self._robot.getPosition()[0]
        self._pos_y = self._robot.getPosition()[1]

    def update_current_theta(self):
        self._theta = self._array_to_euler_angles(self._robot.getOrientation(), False)[
            2
        ]

        self._theta = self._theta % (2 * np.pi)

    def update_ranges(self, ranges):
        self._lidar_distances = ranges

    def follow_boundary(self):
        _obstacles = self._find_obstacles(self._lidar_distances)
        _state = State.BOUNDARY_FOLLOWING

        if not _obstacles:
            _state = State.MOTION_TO_GOAL
            return _state, 0, 0

        _pos = np.array([self._pos_x, self._pos_y])

        if self._bound_pos:
            bound_dists = np.linalg.norm(np.array(self._bound_pos) - _pos, axis=1)

            if not self._check_again and np.max(bound_dists) > self._err:
                self._check_again = 1
            elif self._check_again and any(bound_dists[:-15] <= self._err):
                _state = State.STUCK
                return _state, 0, 0

        _closest_obj = self._choose_object(_obstacles)
        self._bound_pos.append([self._pos_x, self._pos_y])

        _x = _closest_obj[0] + self._pos_x
        _y = _closest_obj[1] + self._pos_y

        _state = State.MOVING

        if self._d_reach <= self._d_followed - self._err:
            _state = State.MOTION_TO_GOAL
            self._bound_pos = np.array([self._pos_x, self._pos_y ])

        return _state, _x, _y

    def move_to_goal(self):
        _x, _y = self._pos_x, self._pos_y

        _obstacles = self._find_obstacles(self._lidar_distances)

        _blocking, _obj = self._check_any_obstacle_ahead(
            _obstacles, self._target_x, self._target_y
        )

        if _blocking:
            _d_reach_old = self._d_reach

            _obj_to_follow = self._choose_object(_obstacles)

            _x, _y = _obj_to_follow[0], _obj_to_follow[1]

            _dist_obj_to_target = np.linalg.norm(
                np.array([self._target_x, self._target_y])
                - np.array([_obj_to_follow[0], _obj_to_follow[1]])
            )
            _dist_robot_to_target = np.linalg.norm(
                np.array([self._pos_x, self._pos_y])
                - np.array([self._target_x, self._target_y])
            )

            if _dist_obj_to_target > _dist_robot_to_target:
                self._d_followed = _d_reach_old
            pass

        else:
            _x, _y = self._target_x, self._target_y

        self._last_motion_vec = [self._pos_x, self._pos_y] - np.array([_x, _y])

        norm_vec = self._last_motion_vec / np.linalg.norm(self._last_motion_vec)
        dot_prod = np.dot(norm_vec, np.array([1, 0]))

        if np.sign(np.arctan(dot_prod) + 2 * np.pi) == 1:
            self._last_motion_ang = np.pi / 2
        else:
            self._last_motion_ang = -np.pi / 2

        return _x, _y, _blocking


class MyRobot:
    def __init__(self, super, target_x, target_y):
        self.robot = super.getFromDef("turtlebot")
        self.super = super
        self._motion_controller = RobotMotionController(self.robot, self.super)
        self._trajectory_planner = RobotTrajectoryPlanner(
            self.robot, target_x, target_y
        )
        self._sensor = self.super.getDevice("LDS-01")
        self._trajectory_planner.set_lidar_info(self._sensor.getMaxRange(), 180)
        self._target_x = target_x
        self._target_y = target_y

        self._follow_x = 0
        self._follow_y = 0

        self._sensor.enable(10)
        self._state = State.MOTION_TO_GOAL

    def _change_state(self, u: State):
        self._state = u
        self._trajectory_planner._bound_pos = []
        self._trajectory_planner._check_again = 0
        print(self._state.name)

    def change_target(self, x, y):
        self._target_x = x
        self._target_y = y

    def run(self):
        distances = self._sensor.getRangeImage()
        self._motion_controller.update_current_xy()
        self._motion_controller.update_current_theta()

        self._trajectory_planner.update_current_xy()
        self._trajectory_planner.update_current_theta()
        self._trajectory_planner.update_ranges(distances)

        match self._state:
            case State.MOTION_TO_GOAL:
                _next_x, _next_y, _change_to_other_state = (
                    self._trajectory_planner.move_to_goal()
                )

                distance_to_target = self._motion_controller.get_distance_to_target(
                    self._target_x, self._target_y
                )

                if distance_to_target == 0:
                    self._change_state(State.REACHED)
                    pass

                self._motion_controller.go_to(_next_x, _next_y)

                if _change_to_other_state:
                    self._change_state(State.BOUNDARY_FOLLOWING)
                    pass

            case State.BOUNDARY_FOLLOWING:
                _next_state, _next_x, _next_y = (
                    self._trajectory_planner.follow_boundary()
                )

                self._follow_x = _next_x
                self._follow_y = _next_y

                self._motion_controller.go_to(_next_x, _next_y)
                self._change_state(_next_state)

            case State.REACHED:
                self._motion_controller.stop()

            case State.MOVING:
                self._motion_controller.go_to(self._follow_x, self._follow_y)

                self._change_state(State.BOUNDARY_FOLLOWING)

            case State.STUCK:
                self._motion_controller.stop()

    def stop(self):
        self._motion_controller.stop()

    def step(self):
        return self.super.step(int(self.super.getBasicTimeStep()))


super = Supervisor()
TARGET_POS = super.getFromDef("target_duck").getPosition()[:2]

robot_k = MyRobot(super, TARGET_POS[0], TARGET_POS[1])
robot_k.stop()

print(robot_k._state.name)
while robot_k.step() != -1:
    robot_k.run()
    pass
