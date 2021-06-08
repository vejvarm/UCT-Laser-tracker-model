import unittest
from construct import Environment, Servo, Laser, Construct

class TestEnvironment(unittest.TestCase):
    env = Environment()
    fov = env.get_fov()

    def test_get_fov(self):
        fov = self.env.get_fov()
        self.assertEqual(fov, self.fov)
        print(fov)

    def test_get_ppm(self):
        pass


class TestServo(unittest.TestCase):

    servo = Servo()
    bounds, default_angle, speed, slippage = servo.get_settings()

    def test_get(self):
        self.assertEqual(self.servo.get_angle(), self.servo._angle)

    def test_set(self):
        angles_to_set = list(range(-20, 200, 10))
        for ang in angles_to_set:
            self.servo.set_angle(ang)
            self.assertEqual(max(min(ang, self.bounds[1]), self.bounds[0]), self.servo.get_angle())

    def test_move_to(self):
        init_angle = self.bounds[0]
        final_angle = 200
        self.servo.set_angle(init_angle)
        for ang in range(init_angle, final_angle, self.speed):
            self.assertEqual(max(min(ang, self.bounds[1]), self.bounds[0]), self.servo.get_angle())
            self.servo.move_to(final_angle)
        self.assertEqual(max(min(ang, self.bounds[1]), self.bounds[0]), self.servo.get_angle())

        init_angle = self.bounds[1]
        final_angle = -20
        self.servo.set_angle(init_angle)
        for ang in range(init_angle, final_angle, -self.speed):
            self.assertEqual(max(min(ang, self.bounds[1]), self.bounds[0]), self.servo.get_angle())
            self.servo.move_to(final_angle)
        self.assertEqual(max(min(ang, self.bounds[1]), self.bounds[0]), self.servo.get_angle())

    def test_wall_position(self):
        angle = 30
        fov = 5

        pixel_distance = self.servo.get_dot_wall_position(angle, fov)

        print(pixel_distance)


class TestLaser(unittest.TestCase):

    laser = Laser()

    def test_move_x_y_tick(self):
        angles_x = list(range(10, 171, 20))
        angles_y = list(range(170, 9, -20))

        print()
        for ax, ay in zip(angles_x, angles_y):
            done = False
            while not done:
                done = self.laser.move_x_y_tick(ax, ay)
            print(f"angles: ({ax}, {ay}) | pos_x: {self.laser.wall_pos_x} | pos_y: {self.laser.wall_pos_y}")