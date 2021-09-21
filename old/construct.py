# class Construct:
#     """
#
#     """
#     __vertical_laser_distance = 0.2  # m
#
#     def __init__(self, lasers=(None, None), visualize=False):
#         """
#
#         :param lasers:
#         :param visualize:
#         """
#         if lasers[0] is None:
#             self._laser_red = Laser()
#         else:
#             self._laser_red = lasers[0]
#         if lasers[1] is None:
#             self._laser_green = Laser()
#         else:
#             self._laser_green = lasers[1]
#
#         self.visualize = visualize
#
#         self.green_pos = (self._laser_green.wall_pos_x, self._laser_green.wall_pos_y)
#         self.red_pos = (self._laser_red.wall_pos_x, self._laser_red.wall_pos_y)
#
#         if self.visualize:
#             self.wall = Wall(blit=True)
#         else:
#             self.wall = None
#
#         self.env = Transformations()
#
#         self.default_path_x, self.default_path_y = generate_path()
#
#         self.state = namedtuple("TimeStep", ("discount", "observation", "reward", "step_type", "done"))
#         self.state.discount = 1.
#         self.state.observation = np.array([self._laser_green.wall_pos_x, self._laser_green.wall_pos_y,
#                                            self._laser_red.wall_pos_x, self._laser_red.wall_pos_y])
#         self.state.reward = self.env.reward(np.expand_dims(self.state.observation, 0))
#         self.state.step_type = 0
#         self.state.done = False
#
#     def reset(self):
#         self.__init__()
#         return self.state
#
#     def step(self, x=None, y=None, speed_restrictions=True):
#         x_red = next(self.default_path_x)
#         y_red = next(self.default_path_y)
#
#         # move red laser based on path from path_gen
#         done = self._laser_red.move_angle_tick(x_red, y_red, speed_restrictions)
#
#         # update laser position indicators
#         self.green_pos = (self._laser_green.wall_pos_x, self._laser_green.wall_pos_y)
#         self.red_pos = (self._laser_red.wall_pos_x, self._laser_red.wall_pos_y)
#
#         # move green laser based on inputs/path from path_gen
#         _ = self._laser_green.move_angle_tick(x, y, speed_restrictions)
#
#         # DONE: reimplement the done indicator!
#         # what we want to return: TimeStep({'discount', 'observation', 'reward', 'step_type'})
#         self.state.discount = 1.
#         self.state.observation = np.array([*self.green_pos, *self.red_pos])
#         self.state.reward = self.env.reward(np.expand_dims(self.state.observation, 0))
#         self.state.step_type = 0
#         self.state.done = done
#         return self.state