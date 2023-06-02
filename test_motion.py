import numpy as np
import math
import matplotlib.pyplot as plt

global linear_vel, angular_vel, dt

linear_vel = 1
angular_vel = 1
dt = 0.01

def motion_model(particle):
    #Implement the motion model to predict next position
    #We will assume a constant velocity + noise approach
    global linear_vel
    global angular_vel

    #Define noise
    mu_xy=0
    sigma_xy=0.001
    mu_theta=0
    sigma_theta=0.001
    noise_x = np.random.normal(mu_xy,sigma_xy)
    noise_y = np.random.normal(mu_xy,sigma_xy)
    noise_theta = np.random.normal(mu_theta,sigma_theta)
    #Define the matrices for motion model
    matrixA=np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
    arrayB=np.array([linear_vel*dt*math.cos(particle['pose'][2]),
                     linear_vel*dt*math.sin(particle['pose'][2]),
                     angular_vel*dt])
    arrayNoise=np.array([noise_x,noise_y,noise_theta])
    #Update the new position
    pose_array=np.array(particle['pose'])#Turns the pose of the particle into an array for matrix multiplication
    new_pose= matrixA @ pose_array + arrayB + arrayNoise
    x,y,theta = new_pose
    particle['pose'] = [x,y,theta]
    #Return the new particle with the new 'pose'
    return particle

p = {
    'pose': [0,0,0],
    'landmarks': [],
    'weight': 1
}
x = np.array([p['pose'][0]])
y = np.array([p['pose'][1]])
for i in range(100):
    p = motion_model(p)
    x = np.append(x, p['pose'][0])
    y = np.append(y, p['pose'][1])

plt.plot(x, y)
plt.show()