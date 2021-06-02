import unittest
from construct import Servo, Laser, Construct


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
        pass
