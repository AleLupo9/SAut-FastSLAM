import numpy as np
from numpy import pi
from math import cos, sin, tan, atan2, sqrt, pow
import matplotlib.pyplot as plt
import json
from celluloid import Camera
import random

def deviation(angle):
    if angle == pi/2 or angle == 2*pi-pi/2:
        angle = angle + pi/100
    return angle

def cone_detection(land, angle, pos):
    a_eff = angle/2
    a_sx = deviation(pos[2]+a_eff)
    a_dx = deviation(pos[2]-a_eff)
    m_sx = tan(a_sx)
    m_dx = tan(a_dx)
    if cos(a_sx) < 0 and cos(a_dx) > 0:
        if land[1] > m_sx*(land[0]-pos[0]) + pos[1] and land[1] > m_dx*(land[0]-pos[0]) + pos[1]:
            return True
        else:
            return False
    elif cos(a_sx) < 0 and cos(a_dx) < 0:
        if land[1] > m_sx*(land[0]-pos[0]) + pos[1] and land[1] < m_dx*(land[0]-pos[0]) + pos[1]:
            return True
        else:
            return False
    elif cos(a_sx) > 0 and cos(a_dx) < 0:
        if land[1] < m_sx*(land[0]-pos[0]) + pos[1] and land[1] < m_dx*(land[0]-pos[0]) + pos[1]:
            return True
        else:
            return False
    elif cos(a_sx) > 0 and cos(a_dx) > 0:
        if land[1] < m_sx*(land[0]-pos[0]) + pos[1] and land[1] > m_dx*(land[0]-pos[0]) + pos[1]:
            return True
        else:
            return False

def proximity(per: float, init_pos, dist: float, var: float, limit: float, lin_vel):
    if per > 1:
        per = 1
    
    if per <= 1 and per > -1:
        pos0 = [dist*cos(init_pos[2]-pi/2), dist*sin(init_pos[2]-pi/2), init_pos[2]]
        rel_pos = [dist*cos(init_pos[2]-pi/2+per*pi), dist*sin(init_pos[2]-pi/2+per*pi), init_pos[2]+per*pi]
        robot[0] = rel_pos[0]-pos0[0]+init_pos[0]
        robot[1] = rel_pos[1]-pos0[1]+init_pos[1]
        robot[2] = rel_pos[2]-pos0[2]+init_pos[2]
    
    if (per == 1 or per == -2) and sqrt(robot[0]**2+robot[1]**2) >= limit:
        per = -2
        robot[0] += lin_vel*cos(robot[2])
        robot[1] += lin_vel*sin(robot[2])
    elif per == -2:
        per = -1
    else:
        per += var
    
    return per, robot


cone_angle = pi/3
n_land = 50
r_l = 4
land_list = []
#landmark = x, y
#land_list[id][coord]

for i in range(n_land):
    theta = random.uniform(0, 1)*2*pi
    r = r_l*sqrt(random.uniform(0, 1))
    landmark = [r*cos(theta), r*sin(theta)]
    land_list.append(landmark)


x_land = [land_list[i][0] for i in range(n_land)]
y_land = [land_list[i][1] for i in range(n_land)]
plt.scatter(x_land, y_land)
# plt.show()
"""
robot = [0, 2.5, pi]
land_det_x=[]
land_det_y=[]
land_not_det=[]
land_not_det_x=[]
land_not_det_y=[]
mu_xy = 0
sigma_xy = 0.2
fig, ax = plt.subplots()
for i in range(n_land):
    if cone_detection(land_list[i], cone_angle, robot):
        corr_x = land_list[i][0]+np.random.normal(mu_xy,sigma_xy)
        corr_y = land_list[i][1]+np.random.normal(mu_xy,sigma_xy)
        land_det_x.append(corr_x)
        land_det_y.append(corr_y)
    else:
        land_not_det_x.append(land_list[i][0])
        land_not_det_y.append(land_list[i][1])
        
ax.scatter(land_det_x, land_det_y, c='blue', label="Landmarks seen by the robot")
ax.scatter(robot[0], robot[1], c='red', label="Robot position")
ax.scatter(land_not_det_x, land_not_det_y, c='green', label="Landmarks not seen by the robot")

ax.legend(bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.show()
"""

total_t = 150
lin_vel = 2*r_l*pi/30
dt = 0.1
max_reach = 3
mu_xy = 0
sigma_xy = 0
data = {}
data2 = {}

robot = [0, 0, 0]
per = -1

fig, ax = plt.subplots()
camera = Camera(fig)

for co in range(int(total_t/dt)):
    xp = []
    yp = []
    # scan
    # f.write("obs:"+str(co+1)+"\ntime:"+str((co+1)*dt)+"\n")
    data["obs"+str(co)] = {"time": (co+1)*dt}
    data2["obs"+str(co)] = {"x": robot[1], "y": robot[0], "theta": robot[2]} 
    co2 = 0
    for i in range(n_land):
        if cone_detection(land_list[i], cone_angle, robot):
            x_rel = land_list[i][0]-robot[0]
            y_rel = land_list[i][1]-robot[1]
            dist = sqrt(pow(x_rel, 2)+pow(y_rel, 2))
            if dist < max_reach:
                angle = atan2(y_rel, x_rel)-robot[2]+pi/2
                x_oriented = dist*cos(angle)
                y_oriented = dist*sin(angle)
                corr_x = x_oriented+np.random.normal(mu_xy,sigma_xy)
                corr_y = y_oriented+np.random.normal(mu_xy,sigma_xy)
                # f.write("id:"+str(i)+"\nx:"+str(corr_x)+"\ny:"+str(corr_y)+"\nerr:"+str(sigma_xy)+"\n")
                data["obs"+str(co)]["land"+str(co2)] = {"id": i, "x": corr_x, "y": corr_y, "err": sigma_xy}
                
                xp.append(land_list[i][0]) # land_list[i][0])
                yp.append(land_list[i][1]) # land_list[i][1])
                co2 += 1
    # f.write("\n")

    # movement
    a_vel = random.uniform(-2*pi, 2*pi)
    if sqrt(robot[0]**2+robot[1]**2) > r_l*0.9 and per == -1:
        init_pos = [0, 0, 0]
        for i in range(len(robot)):
            init_pos[i] = robot[i]
        per = 0
        print(init_pos)
        distt = random.uniform(0.1, 0.2)
        var = lin_vel*dt/(2*pi*distt)
        per = var
    
    if per == -1:
        robot[0] += lin_vel*dt*cos(robot[2]+a_vel*dt/2)
        robot[1] += lin_vel*dt*sin(robot[2]+a_vel*dt/2)
        robot[2] += a_vel*dt/2
        ax.scatter(robot[0], robot[1], c='red')
    else:
        per, robot = proximity(per, init_pos, distt, var, r_l*0.8, lin_vel*dt)
        #print(robot)
        #print(per)
        ax.scatter(robot[0], robot[1], c='green')
    
    if robot[2] >= 2*pi:
        robot[2] -= 2*pi
    elif robot[2] < 0:
        robot[2] += 2*pi

    #ax.scatter(robot[0], robot[1], c='red')
    ax.scatter(xp, yp, c='blue')
    camera.snap()

tot_data = [data, data2]
with open("simulation_com.json", "w") as file_json:
    json.dump(tot_data, file_json)

animation = camera.animate(interval=dt*100)  # Intervallo di tempo tra i frame (in millisecondi)
plt.show()