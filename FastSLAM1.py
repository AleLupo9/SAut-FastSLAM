import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.linalg as expm
import rosbag # Assuming we have rosbag installed
#modify??

global precision, base_weight
precision = 0.001
base_weight = 1

def read_rosbag_data(filename):
    bag=rosbag.Bag(filename)
    #Extract relevant data from the bag
    #These should be the measurements (aruco detections)
    #Return the extracted data as arrays or lists

def motion_prediction(particle, linear_vel, angular_vel, dt):
    #Implement the motion model to predict next position
    #Probably we don't need controls "u" because we do not have odometry -> assume constant velocity?
    displacement=np.array([linear_vel*dt*math.cos(particle['pose'][3]),
                     linear_vel*dt*math.sin(particle['pose'][3]),
                     angular_vel*dt])

    #Update the new position
    pose_array=np.array[particle['pose']] #Turns the pose of the particle into an array for matrix multiplication
    new_pose= pose_array + displacement # + NOISE????? (imo no)
    x,y,theta = new_pose
    particle['pose'] = [x,y,theta]
    #Return the new particle with the new 'pose'
    return particle


def actual_displacement(x, y, theta):
    z = np.array([x,y,theta])
    return z


def predicted_lndmrk_position(particle, det_lndmrk):
    dx = particle['landmarks'][det_lndmrk]['mu'][1]-particle['pose'][1]
    dy = particle['landmarks'][det_lndmrk]['mu'][2]-particle['pose'][2]
    x_lndmrk, y_lndmrk = movement_function(dx, dy, particle['pose'][3])
    return x_lndmrk, y_lndmrk

def movement_function(dx, dy, theta):
    x = math.sqrt(dx**2+dy**2)*math.cos(math.atan2(dy, dx)-theta)
    y = math.sqrt(dx**2+dy**2)*math.sin(math.atan2(dy, dx)-theta)
    return x, y

def inverse_movement(particle, z):
    theta = math.atan2(z[2], z[1])
    mu_x = particle['pose'][1]+math.sqrt(z[1]**2+z[2]**2)*math.cos(theta+particle['pose'][3])
    mu_y = particle['pose'][2]+math.sqrt(z[1]**2+z[2]**2)*math.cos(theta+particle['pose'][3])
    return mu_x, mu_y

def jacobian(dx, dy, theta):
    x,y = movement_function(dx, dy, theta)
    x_x,y_x = movement_function(dx+precision, dy, theta)
    x_y,y_y = movement_function(dx, dy+precision, theta)
    H = np.array([(x_x-x)/precision, (x_y-x)/precision],
                 [(y_x-y)/precision, (y_y-y)/precision])
    return H


def measurement_model(z,x):
    #Calculates the expected measurement given the current estimate of the landmark

    #Return the expected measurement to be used in EKF update probably
    return()

def initialize_lndmrk(particle, landmark_id, z, err):
    mu = inverse_movement(particle, z)
    H = jacobian(mu[1]-particle['pose'][1], mu[2]-particle['pose'][2], particle['pose'][3])
    Q = np.eye(2)*err
    var = np.linalg.inv(H)@Q@np.transpose(np.linalg.inv(H))
    w = base_weight
    return mu, var, w
    
def update_lndmrk(particle, landmark_id, z, err)
    mu_old = particle['landmarks'][landmark_id]['mu']
    sigma_old = particle['landmarks'][landmark_id]['sigma']
    x = particle['pose']
    z_hat = movement_function(mu_old[1]-x[1], mu_old[2]-x[2])
    H = jacobian(mu_old[1]-x[1], mu_old[2]-x[2], x[3])
    Q = H@sigma_old@np.transpose(H) + np.eye(2)*err
    K = sigma_old@np.transpose(H)@np.linalg.inv(Q)
    mu = mu_old + K*(z-z_hat)
    sigma = (np.eye(2)-K@H)*sigma_old
    w = np.linalg.matrix_power(np.abs(2*math.pi*Q), -0.5)


def is_landmark_seen(particle, landmark_id):
    for landmark in particle['landmarks']:
        if landmark['id'] == landmark_id:
            return True  # Landmark already seen in the particle
    return False  # Landmark is new

def ekf_update_landmark(mu, sigma, z, Q):
    # Implements the EKF update for a single landmark
    # mu: Mean of the landmark estimate
    # sigma: Covariance matrix of the landmark estimate
    # z: Measurement of the landmark
    # Q: Measurement noise covariance matrix
    return()

def resample_particles(particles, weights):
    return()
    #return the resampled particles

def fastslam_kc(ParticleSet,measurements):
    for k in range(num_particles):
        #Sample new pose -> Motion Model
        ParticleSet[k]=motion_model(ParticleSet[k])
        #Loop in the number of observations done in each instant 
        #(there might be a possibility that the robot does multiple observations at the same instant)
        for i in range(measurements):
            landmark_id=measurements[i][0]
            
    
    return ParticleSet, pose,landmarks #for each t

#Some parameters to define, such as timestep, linear_vel and angular_vel
dt=0.1 #(s)
linear_vel=0.5 #(m/s)
angular_vel=0.5 #(rad/s)

#Load data from rosbag
filename='path_to_rosbag_file'
bag = read_rosbag_data(filename)

#Define the range for each dimension
x_min=0
x_max=20
y_min=0
y_max=20
theta_min=-np.pi
theta_max= np.pi


#Initiate the ParticleSet:
num_particles=100
num_landmarks=5 #Put here the number of the landmarks. We should know their id and it should be by order.
particle_set=[] #Holds each particle. Each particle is a dictionary that should have 'pose' and 'landmarks'.
                #The 'pose' section has a list of 3 variables (x,y,theta)
                #The landmarks section has, for each landmark, a list for the 'mu' and a matrix 'sigma'

#We assume a random uniform distribution for the robot's pose in the particles. 
#We don't initialize mean values nor covariances for the landmarks because the robot has not yet detected them
for i in range(num_particles):
    x=random.uniform(x_min,x_max)
    y=random.uniform(y_min,y_max)
    theta=random.uniform((theta_min,theta_max))
    new_particle={
        'pose': [x,y,theta],
        'landmarks': []
    }
    
    #Loop for each landmark
    for j in range(num_landmarks):
        new_landmark={
            'id': j, #Assuming we use the ids in order, i.e, if we use 5 markers, we are using those which have id=0,1,2,3,4
            'mu': [], #mean vector
            'sigma': [] #covariance matrix 2X2
        }
        new_particle['landmarks'].append(new_landmark)
    
    #Add the new_particle to the particle_set variable
    particle_set.append(new_particle)
    


#Iterate over the messages in the bag file
for topic, msg, t in bag.read_messages():
    measurements=[]
    if topic == '/fiducial_transforms':
        for fiducial in msg.fiducials:
            fiducial_id = fiducial.fiducial_id
            translation_x = fiducial.transform.translation.x
            translation_y = fiducial.transform.translation.y
            #translation_z = fiducial.transform.translation.z

            #Add the landmark measurements to a variable. In this case we are not discarding the possibility of the robot detecting more than one aruco marker
            measurements.append([fiducial_id,translation_x,translation_y])


    particle_set, robot_pose, landmark_pose = fastslam_kc(particle_set, num_particles, measurements)