import numpy as np
import math
import random
import matplotlib.pyplot as plt
import json

def motion_model(particle):
    #Implement the motion model to predict next position
    #We will assume a constant velocity + noise approach
    global linear_vel
    global angular_vel
    #Define noise
    per_xy = 0
    per_theta = 0
    mu_xy=0
    sigma_x=abs(linear_vel*dt*per_xy)
    sigma_y=abs(linear_vel*dt*per_xy)
    mu_theta=0
    sigma_theta=abs(angular_vel*dt*per_theta)
    noise_x = np.random.normal(mu_xy,sigma_x)
    noise_y = np.random.normal(mu_xy,sigma_y)
    noise_theta=np.random.normal(mu_theta,sigma_theta)
    #Define the matrices for motion model
    matrixA=np.array([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
    arrayB=np.array([linear_vel*dt*math.cos(particle['pose'][2]+math.pi/200),
                     linear_vel*dt*math.sin(particle['pose'][2]+math.pi/200),
                     angular_vel*dt])
    arrayNoise=np.array([noise_x,noise_y,noise_theta])
    #Update the new position
    pose_array=np.array(particle['pose'])#Turns the pose of the particle into an array for matrix multiplication
    new_pose= matrixA @ pose_array + arrayB + arrayNoise
    x,y,theta = new_pose
    particle['pose'] = [x,y,theta]
    #Return the new particle with the new 'pose'
    return particle

def h_inverse(particle,z):
    alpha=math.atan2(z[2],z[1])
    d=math.sqrt(z[1]**2 + z[2]**2)
    mu_x=particle['pose'][0] + d*math.cos(alpha + particle['pose'][2] - math.pi/2)
    mu_y=particle['pose'][1] + d*math.sin(alpha + particle['pose'][2] - math.pi/2)
    return mu_x, mu_y

def h_function(x,y,theta,mu_x,mu_y):
    dx=mu_x-x
    dy=mu_y-y
    d=math.sqrt(dx**2+dy**2)
    z_x=d*math.cos(math.pi/2 - theta + math.atan2(dy,dx))
    z_y=d*math.sin(math.pi/2 - theta + math.atan2(dy,dx))
    return z_x, z_y

def jacobian(x,y,theta,mu_x,mu_y):
    #Define elements to go inside the matrix -> function values and small deviations
    z_x,z_y=h_function(x,y,theta,mu_x,mu_y)
    x_x,y_x=h_function(x+precision,y,theta,mu_x,mu_y)
    x_y,y_y=h_function(x,y+precision,theta,mu_x,mu_y)

    #Define matrix
    H = np.array([[(x_x-z_x)/precision,(x_y-z_x)/precision],
               [(y_x-z_y)/precision,(y_y-z_y)/precision]])
    
    return H
    
def is_landmark_seen(particle, landmark_id):
    landmarks = particle['landmarks']
    if landmarks:
        return any(landmark['id'] == landmark_id for landmark in landmarks)
    else:
        return False

def initialize_landmark(particle,z,err,landmark_id):
    new_landmark=[]

    mu=h_inverse(particle,z)
    H=jacobian(particle['pose'][0],particle['pose'][1],particle['pose'][2],mu[0],mu[1])
    Q=np.eye(2)*err
    sigma= np.linalg.inv(H) @ Q @np.transpose(np.linalg.inv(H))
    new_landmark={
        'id':landmark_id,
        'mu':mu,
        'sigma':sigma,
     }
    return new_landmark
    
def update_landmark(particle, landmark_id, z, err):
    #We just want to alter the landmark whose id matches the landmark_id
    landmarks=particle['landmarks']
    for i,landmark in enumerate(landmarks):
        if landmark['id']==landmark_id:
            index=i
            break
    
    mu_old=particle['landmarks'][index]['mu']
    sigma_old=particle['landmarks'][index]['sigma']
    z_hat=h_function(particle['pose'][0],particle['pose'][1],particle['pose'][2],mu_old[0],mu_old[1])
    H = jacobian(particle['pose'][0],particle['pose'][1],particle['pose'][2],mu_old[0],mu_old[1])
    Q=H @ sigma_old @ np.transpose(H) + np.eye(2)*err
    K= sigma_old @ np.transpose(H) @ np.linalg.inv(Q)
    # print(K)
    #Create array (our z contains (id,landmark_x, landmark_y) whereas our z_hat contains (landmark_x,landmark_y))
    z_deviation=np.array([z[1]-z_hat[0],z[2]-z_hat[1]])
    # print(z_deviation)
    mu_new = mu_old + K @ z_deviation
    sigma_new= (np.eye(2) - K@H) @ sigma_old
    Qdet=np.linalg.det(2*np.pi*Q)
    new_weight=Qdet**(-1/2)*np.exp(-1/2*np.transpose(z_deviation) @ np.linalg.inv(Q) @ z_deviation)
    # print(f"NEW WEIGHT {new_weight} {Qdet} {z_deviation} {Q}")

    #Apply the new values to the respective landmark and the new weight to the particle
    particle['weight'] = new_weight
    particle['landmarks'][index]['mu']=mu_new
    particle['landmarks'][index]['sigma']=sigma_new

    return(particle)

def normalize_weights(ParticleSet_var):
    #Normalize the weights
    total_weight=0
    for particle in ParticleSet_var:
        total_weight=total_weight+float(particle['weight'])

    for particle in ParticleSet_var:
        particle['weight'] = float(particle['weight']) / total_weight
    
    total_weight=0
    for particle in ParticleSet_var:
        total_weight=total_weight+float(particle['weight'])

    return (ParticleSet_var)

def resample_particles(ParticleSet,num_particles):
    #THE WEIGHTS IN THE PARTICLES SHOULD BE NORMALIZED
    weights=[]
    indices=[]
    num_landmarks = len(ParticleSet[0]['landmarks'])
    for i in range(num_particles):
        weights.append(ParticleSet[i]['weight'])
    indices=np.arange(num_particles)
    resampled_indices=np.random.choice(indices,size=num_particles,p=weights)
    new_particle_set=[]
    
    for i in resampled_indices:
        x=ParticleSet[i]['pose'][0]
        y=ParticleSet[i]['pose'][1]
        theta=ParticleSet[i]['pose'][2]
        new_particle={
        'pose': [x,y,theta],
        'landmarks': [],
        'weight': base_weight}
        #Loop for each landmark
        for j in range(num_landmarks):
           id=ParticleSet[i]['landmarks'][j]['id']
           mu=ParticleSet[i]['landmarks'][j]['mu']
           sigma=ParticleSet[i]['landmarks'][j]['sigma']
           new_landmark={
            'id': id, #Assuming we use the ids in order, i.e, if we use 5 markers, we are using those which have id=0,1,2,3,4
            'mu': mu,
            'sigma': sigma
            }
           new_particle['landmarks'].append(new_landmark)
           
        #Add the new_particle to the new_particle_set variable
        new_particle_set.append(new_particle)

    return (new_particle_set)

def retrieve_landmark_positions(ParticleSet,weights):
    num_landmarks=len(ParticleSet[0]['landmarks']) #Every particle has ALWAYS the same number of landmarks
    landmark_positions=[[] for _ in range(num_landmarks)] #Creates a list of empty lists. Each of these lists correspond to a landmark

    for particle in ParticleSet:
        landmarks=particle['landmarks']
        for i,landmark in enumerate(landmarks):
            landmark_mean=landmark['mu']
            landmark_positions[i].append(landmark_mean)

    weighted_landmark_positions=[]
    for landmark in landmark_positions:
        landmark=np.array(landmark)
        weighted_mean=np.average(landmark,axis=0,weights=weights)
        weighted_landmark_positions.append(weighted_mean.tolist())

    return weighted_landmark_positions

def fastslam_kc(ParticleSet,num_particles,measurements):
    for k in range(num_particles):
        #Sample new pose -> Motion Model
        ParticleSet[k]=motion_model(ParticleSet[k])
        #Loop in the number of observations done in each instant 
        #(there might be a possibility that the robot does multiple observations at the same instant)
        for i in range(len(measurements)):
            landmark_id=measurements[i][0]
            #See if landmark as been seen
            if not is_landmark_seen(ParticleSet[k],landmark_id):
                new_landmark=[]
                new_landmark=initialize_landmark(ParticleSet[k],measurements[i],err,landmark_id)
                ParticleSet[k]['landmarks'].append(new_landmark)
                ParticleSet[k]['weight'] = base_weight
            else:
                ParticleSet[k]=update_landmark(ParticleSet[k],landmark_id,measurements[i],err)

    ParticleSet=normalize_weights(ParticleSet)

    #Take robot's position and landmark position
    weights=np.array([particle['weight'] for particle in ParticleSet])
    poses=np.array([particle['pose'] for particle in ParticleSet])
    pose_estimate=np.average(poses,axis=0,weights=weights)
    landmarks_estimate=retrieve_landmark_positions(ParticleSet,weights)

    #Resample particles
    ParticleSet=resample_particles(ParticleSet,num_particles)
    return ParticleSet,pose_estimate,landmarks_estimate #for each t

def plot_robot_pose_and_landmarks(robot_positions, landmarks_pose):

    #Extract correctly the robot's positions (over all time -> Path)
    robot_x=[robot_positions[i][0] for i in range(len(robot_positions))]
    robot_y=[robot_positions[i][1] for i in range(len(robot_positions))]

    #Extract correctly the landmark positions(at last iteration -> Final landmark positions)
    landmark_x=[landmark[0] for landmark in landmarks_pose]
    landmark_y=[landmark[1] for landmark in landmarks_pose]

    #Plot the robot's position
    plt.scatter(robot_x,robot_y,color='blue', label='Robot Path')

    #Plot the landmarks positions
    plt.scatter(landmark_x,landmark_y,color='red',label='Landmarks')

    #Add labels
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('SLAM.png')
    plt.clf()

#Some parameters to define, such as timestep, linear_vel and angular_vel
global n_turns
global num_particles
n_turns = 5
r = 2
turn_t = 50
angular_vel=2*math.pi/turn_t
linear_vel=r*math.sqrt(2*(1-math.cos(angular_vel*0.1)))/0.1
precision=0.01
err=0.05

#Define the range for each dimension
x_min=-0
x_max=0
y_min=-0
y_max=0
theta_min=0 # math.pi/2-math.pi/12
theta_max=0 #math.pi/2+math.pi/12

#Initiate the ParticleSet:
num_particles=50
base_weight=1/num_particles
#num_landmarks=5 #Put here the number of the landmarks. We should know their id and it should be by order.
ParticleSet=[] #Holds each particle. Each particle is a dictionary that should have 'pose' and 'landmarks'.
                #The 'pose' section has a list of 3 variables (x,y,theta)
                #The landmarks section has, for each landmark, a list for the 'mu' and a matrix 'sigma'

#We assume a random uniform distribution for the robot's pose in the particles. 
#We don't initialize mean values nor covariances for the landmarks because the robot has not yet detected them
for i in range(num_particles):
    x=random.uniform(x_min,x_max)
    y=random.uniform(y_min,y_max)
    theta=random.uniform(theta_min,theta_max)
    new_particle={
        'pose': [x,y,theta],
        'landmarks': [],
        'weight': base_weight}
    
    #Add the new_particle to the particle_set variable
    ParticleSet.append(new_particle)
    

#Create list for all the positions of the robot
robot_positions=[]
land_int = []

with open("simulation.json", "r") as file_json:
    data = json.load(file_json)

old_time = -1

#Iterate over the messages in the bag file
for i in range(len(data)):
    measurements=[]

    if old_time == -1:
        dt = 1
    else:
        current_time = data["obs"+str(i)]["time"]
        dt = current_time-old_time
        old_time = current_time

    for j in range(len(data["obs"+str(i)])-1):
        print("postazione\n"+str(i)+"\n"+str(j))
        fiducial_id = data["obs"+str(i)]["land"+str(j)]["id"]
        translation_x = data["obs"+str(i)]["land"+str(j)]["x"]
        translation_y = data["obs"+str(i)]["land"+str(j)]["y"]

        #Add the landmark measurements to a variable. In this case we are not discarding the possibility of the robot detecting more than one aruco marker
        measurements.append([fiducial_id,translation_x,translation_y])
        length=len(measurements)

    ParticleSet,pose_estimate, landmarks_pose = fastslam_kc(ParticleSet,num_particles, measurements)
    land_int.append(landmarks_pose[3])
    robot_positions.append(pose_estimate)

plot_robot_pose_and_landmarks(robot_positions,landmarks_pose)

plt.figure(1)
plt.scatter([land_int[i][0] for i in range(len(land_int))], [land_int[i][1] for i in range(len(land_int))])
labels = [str(i) for i in range(len(land_int))]
for j in range(len(land_int)):
    plt.text(land_int[j][0], land_int[j][1], labels[j], ha='center', va='bottom')
plt.show()