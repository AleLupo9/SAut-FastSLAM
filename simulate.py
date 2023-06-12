import numpy as np
from numpy import pi
from math import cos, sin, tan, atan2, sqrt, pow
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

cone_angle = pi/3
n_land = 30
r_l = 4
land_list = []
# landmark = x, y
# land_list[id][coord]

for i in range(n_land):
    landmark = [r_l*cos(2*pi*i/n_land), r_l*sin(2*pi*i/n_land)]
    land_list.append(landmark)

"""
#x_land = [land_list[i][0] for i in range(n_land)]
#y_land = [land_list[i][1] for i in range(n_land)]
#plt.scatter(x_land, y_land)
#plt.show()
# robot = [x, y, theta]
"""
"""
robot = [0, 2, pi]

mu_xy = 0
sigma_xy = 0.05

for i in range(n_land):
    if cone_detection(land_list[i], cone_angle, robot):
        corr_x = land_list[i][0]+np.random.normal(mu_xy,sigma_xy)
        corr_y = land_list[i][1]+np.random.normal(mu_xy,sigma_xy)
        plt.scatter(corr_x, corr_y, c='blue')
    else:
        plt.scatter(land_list[i][0], land_list[i][1], c='green')
plt.scatter(robot[0], robot[1], c='red')
plt.show()


"""
n_turns = 10
turn_t = 30
r_r = 3
a_vel = 2*pi/turn_t
dt = 1
max_reach = 3
mu_xy = 0
sigma_xy = 0
data = {}

robot = [r_r, 0, 0]
a_abs = atan2(robot[1], robot[0])
robot[2] = a_abs + pi/2

fig, ax = plt.subplots()
camera = Camera(fig)

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
                # f.write("id:"+str(i)+"\nx:"+str(corr_x)+"\ny:"+str(corr_y)+"\nerr:"+str(sigma_xy)+"\n")
                data["obs"+str(co)]["land"+str(co2)] = {"id": i, "x": corr_x, "y": corr_y, "err": sigma_xy}
                
                xp.append(corr_x) # land_list[i][0])
                yp.append(corr_y) # land_list[i][1])
                co2 += 1
    # f.write("\n")

    # movement
    a_abs += a_vel*dt
    if a_abs >= 2*pi:
        a_abs -= 2*pi
    robot[0] = r_r*cos(a_abs)
    robot[1] = r_r*sin(a_abs)
    robot[2] = a_abs + pi/2

    ax.scatter(0, 0, c='red')
    ax.scatter(xp, yp, c='blue')
    camera.snap()

# f.close()
with open("simulation.json", "w") as file_json:
    json.dump(data, file_json)

animation = camera.animate(interval=dt*100)  # Intervallo di tempo tra i frame (in millisecondi)
plt.show()