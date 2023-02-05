from numpy import array, concatenate, cos, dot, reshape, sin, zeros
from core.dynamics import RoboticDynamics


class Dubin_Car(RoboticDynamics): # Creating a dubins car model
    def __init__(self, mass, l, g=9.81):
        RoboticDynamics.__init__(self, 4, 2) # Initialize robot dynamics with 4 states X,Y,v,theta
        self.params = mass, l, g

    # def D(self, q):
    #     m_c, m_p, l, _ = self.params
    #     _, theta = q
    #     return array([[m_c + m_p, m_p * l * cos(theta)], [m_p * l * cos(theta), m_p * (l ** 2)]])
    #
    # def C(self, q, q_dot):
    #     _, m_p, l, _ = self.params
    #     _, theta = q
    #     _, theta_dot = q_dot
    #     return array([[0, -m_p * l * theta_dot * sin(theta)], [0, 0]])
    #
    # def U(self, q):
    #     _, m_p, l, g = self.params
    #     _, theta = q
    #     return m_p * g * l * cos(theta)
    #
    # def G(self, q):
    #     _, m_p, l, g = self.params
    #     _, theta = q
    #     return array([0, -m_p * g * l * sin(theta)])
    #
    # def B(self, q):
    #     return array([[1], [0]])
