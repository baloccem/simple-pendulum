#!/usr/bin/env python

'''
Simple pendulum calculation. 
Author: Emilio Balocchi
'''

import re,math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations

class Pendulum():
    t = 0
    g               =  -9.916          # Gravity acceleration (m/s^2 )
    alfa            = -math.pi/2.0


    def __init__(self, theta_prev, ang_vel_prev, ang_accel_prev,m,l,dt):
        self.theta_prev     = theta_prev
        self.ang_vel_prev   = ang_vel_prev
        self.ang_accel_prev = ang_accel_prev
        self.m              = m
        self.l              = l
        self.dt             = dt

    def time(self):
        self.t = self.t + self.dt  

    def position(self):
        self.theta = self.theta_prev +self.ang_vel_prev*self.dt +\
             0.5*self.ang_accel_prev*self.dt**2 

    def velocity(self):
        self.ang_vel = self.ang_accel_prev*self.dt + self.ang_vel_prev
    
    def tangential_force(self):
        self.Ft = -self.m*self.g*math.sin(self.theta) 

    def aceleration(self):
        self.ang_accel = self.Ft/self.m/self.l
    
    def polar_to_cartesian(self):
        self.x_loc = self.l*math.cos(self.theta)
        self.y_loc = self.l*math.sin(self.theta)

    def global_coordinates(self):
        self.local_coord = np.array([self.x_loc,self.y_loc])
        self.transf_matr =np.array([[math.cos(self.alfa),-math.sin(self.alfa)],\
                                [math.sin(self.alfa),math.cos(self.alfa)]])
        self.global_coord = self.local_coord.dot(self.transf_matr)
        self.x_glob = self.global_coord[0]
        self.y_glob = self.global_coord[1]
        return self.x_glob, self.y_glob
    
    def assign_previous_values(self):
        self.theta_prev     = self.theta
        self.ang_vel_prev   = self.ang_vel
        self.ang_accel_prev = self.ang_accel
    

    def __str__(self):
        return '{} , {}'.format(self.x_glob,self.y_glob)

    def step(self):

        p.position()
        p.velocity()
        p.tangential_force()
        p.aceleration()
        p.polar_to_cartesian()
        x,y = p.global_coordinates()
        x_data.append(x)
        y_data.append(y)
        p.assign_previous_values()
        p.time()

if __name__ == '__main__':

    x_data = []
    y_data = []

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-3, 3), ylim=(-3, 3))
    line, = ax.plot([], [], 'o-', lw=2, ms = 10)

    p = Pendulum(math.pi/2+0.1,0,0,1,2,0.007)

    def animation_frame(i):

        global p
        p.step()
        thisx = [0, x_data[i]]
        thisy = [0, y_data[i]]
        line.set_data(thisx, thisy)

        return line,

    ani = animation.FuncAnimation(fig, animation_frame, frames=2600,
                                interval=2, blit=True)
    plt.show()