import numpy as np
from numpy import pi
from math import cos, sin, tan, atan2, sqrt, pow, floor
import matplotlib.pyplot as plt
import json
from celluloid import Camera

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

cone_angle = 2*pi/5
n_land = 30
r_l = 4
land_list = []

for i in range(n_land):
    landmark = [r_l*cos(2*pi*i/n_land), r_l*sin(2*pi*i/n_land)]
    land_list.append(landmark)


n_turns = 5
t_s = 10
side = 3.8*sqrt(2)
turn_t = 4*t_s
l_vel = side/t_s
dt = 0.1
max_reach = 3
mu_xy = 0
sigma_xy = 0
data = {}

robot = [-side/2, -side/2, 0]
per = 0
pos = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
qu = -1

fig, ax = plt.subplots()
# camera = Camera(fig)

for co in range(int(turn_t*n_turns/dt)):
    xp = []
    yp = []
    # scan
    # f.write("obs:"+str(co+1)+"\ntime:"+str((co+1)*dt)+"\n")
    data["obs"+str(co)] = {"time": (co+1)*dt}
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
                data["obs"+str(co)]["land"+str(co2)] = {"id": i, "x": corr_x, "y": corr_y, "err": sigma_xy}
                
                xp.append(land_list[i][0]) # land_list[i][0])
                yp.append(land_list[i][1]) # land_list[i][1])
                co2 += 1

    ax.scatter(robot[0], robot[1], c='red')
    # ax.text(robot[0], robot[1], str(co), ha='center', va='bottom')
    ax.scatter(xp, yp, c='blue')

    # movement
    var = dt/t_s
    if abs(robot[0]) > side/2 or abs(robot[1]) > side/2:
        qu += 1
        print(qu)
        if qu == 4:
            qu = 0

    if qu == 0:
        robot[0] += side*var
        robot[1] = -side/2
        robot[2] = 0
    elif qu == 1:
        robot[0] = side/2
        robot[1] += side*var
        robot[2] = pi/2
    elif qu == 2:
        robot[0] -= side*var
        robot[1] = side/2
        robot[2] = pi
    else:
        robot[0] = -side/2
        robot[1] -= side*var
        robot[2] = 3*pi/2

    # camera.snap()


with open("simulation_square_strict.json", "w") as file_json:
    json.dump(data, file_json)

#ax.scatter([land_list[i][0] for i in range(len(land_list))], [land_list[i][1] for i in range(len(land_list))], c="blue")
#animation = camera.animate(interval=dt*100)  # Intervallo di tempo tra i frame (in millisecondi)
plt.show()