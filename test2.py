import matplotlib.pyplot as plt
import math
import random
import numpy as np
import relocation_task_planner as TP
import scipy.spatial
import VFHplus_mobile as vfh
from matplotlib.patches import Rectangle

N_SAMPLE = 2000  # number of sample_points
N_KNN = 10  # number of edge from one sampled point
MAX_EDGE_LEN = 0.05  # [m] Maximum edge length

show_animation = True

box_WH = [0.2, 0.08]  # rectangle width-height
box_Theta = random.randrange(-150, 85)  # anti-clockwise, rectangle angle
box_BL = [0.1, -0.2]  # rectangle bottom-left point
rr = 0.04  # robot radius
ro = 0.04  # object radius
rt = 0.05  # target radius
Flag = False
Nonprehensile_Flag = False


class Node:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)



class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN
        inp: input data, single frame or multi frame
        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index

def PRM_planning(obj, radius,idx, ox, oy):
    sx = obj[idx][0]
    sy = obj[idx][1]
    print("start: ", sx, sy)
    obkdtree = KDTree(np.vstack((ox, oy)).T)
    sample_x, sample_y = sample_points(sx, sy, ox, oy, obj, radius, idx, obkdtree)
    road_map = generate_roadmap(sample_x, sample_y, radius[idx], obkdtree)
    gx, gy = determine_candidate(road_map, sample_x, sample_y)
    sample_x.append(gx)
    sample_y.append(gy)

    if show_animation:
        plt.plot(sample_x, sample_y, ".b", markersize=3)
        plt.plot(gx, gy, "^c")
    rx, ry = dijkstra_planning(obj, radius, idx, sx, sy, gx, gy, road_map, sample_x, sample_y)
    return rx, ry

def sample_points(sx, sy, ox, oy, obj, rad, idx, obkdtree):
    maxx = max(ox)
    maxy = max(oy)
    minx = min(ox)
    miny = min(oy)

    sample_x, sample_y = [], []



    while len(sample_x) <= N_SAMPLE:
        tx = (random.random() * (maxx - minx)) + minx
        ty = (random.random() * (maxy - miny)) + miny

        if Sample_is_collision(tx, ty, obj, rad, idx, rad[idx]):
            continue
        index, dist = obkdtree.search(np.array([tx, ty]).reshape(2, 1))
        sample_x.append(tx)
        sample_y.append(ty)

    sample_x.append(sx)
    sample_y.append(sy)

    return sample_x, sample_y

def generate_roadmap(sample_x, sample_y, rr, obkdtree):
    """
    Road map generation
    sample_x: [m] x positions of sampled points
    sample_y: [m] y positions of sampled points
    rr: Robot Radius[m]
    obkdtree: KDTree object of obstacles
    """

    road_map = []
    nsample = len(sample_x)
    skdtree = KDTree(np.vstack((sample_x, sample_y)).T)

    for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):

        index, dists = skdtree.search(
            np.array([ix, iy]).reshape(2, 1), k=nsample)
        inds = index[0]
        edge_id = []
        #  print(index)

        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]

            if not is_collision(ix, iy, nx, ny, rr, obkdtree):
                edge_id.append(inds[ii])

            if len(edge_id) >= N_KNN:
                break

        road_map.append(edge_id)

    #plot_road_map(road_map, sample_x, sample_y)

    return road_map

def Sample_is_collision(tx, ty, obj, rad, idx, r_curr):
    pop_obj = obj.pop(idx)
    pop_rad = rad.pop(idx)
    d = []
    for i in range(len(obj)):
        dx = tx - obj[i][0]
        dy = ty - obj[i][1]
        d.append(math.hypot(dx, dy))

        if d[i] < r_curr + rad[i]:
            obj.insert(idx, pop_obj)
            rad.insert(idx, pop_rad)
            return True
    obj.insert(idx, pop_obj)
    rad.insert(idx, pop_rad)

    return False

def is_collision(sx, sy, gx, gy, rr, okdtree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.hypot(dx, dy)

    if d >= MAX_EDGE_LEN:
        return True

    D = rr
    nstep = round(d / D)

    for i in range(nstep):
        idxs, dist = okdtree.search(np.array([x, y]).reshape(2, 1))
        if dist[0] <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    idxs, dist = okdtree.search(np.array([gx, gy]).reshape(2, 1))
    if dist[0] <= rr:
        return True  # collision

    return False  # OK

def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")

def determine_candidate(road_map, sample_x, sample_y):
    gx = 0.3
    gy = -0.15
    return gx, gy

def dijkstra_planning(obj, rad, idx, sx, sy, gx, gy, road_map, sample_x, sample_y):
    """
    sx: start x position [m]
    sy: start y position [m]
    gx: goal x position [m]
    gy: goal y position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    rr: robot radius [m]
    road_map: ??? [m]
    sample_x: ??? [m]
    sample_y: ??? [m]
    @return: Two lists of path coordinates ([x1, x2, ...], [y1, y2, ...]), empty list when no path was found
    """

    nstart = Node(sx, sy, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    path_found = True

    while True:
        if not openset:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]
        #current point - check VFH+
        #if VFH+ confirms the effectiveness of the point, then goal_pose is generated.
        # gx, gy = determine_candidate(current.x, current.y)
        # _,_,_,VFH,_ = vfh.influence(len(obj), obj[idx], [obj[k] for k in range(len(obj)-1)],
        #                             [current.x, current.y], 0.6403, rad[k], rad[idx], 0)
        # if VFH:
        #     gx = current.x
        #     gy = current.y

        # show graph
        if show_animation and len(closedset.keys()) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.hypot(dx, dy)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    if path_found is False:
        return [], []

    # generate final course
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind

    return rx, ry


def myrange(start, end, step):
    r = start
    while(r<end):
        yield r
        r += step

def calc_circumscribed_circle():
    """
    R_cirmcumscribed : circumscribed circle
    center_tilted : center point of the circumscribed circle
    """
    center_point = [box_BL[0] + (1/2)*box_WH[0], box_BL[1] + (1/2)*box_WH[1]]

    delta = math.atan2(box_WH[1], box_WH[0])
    # Rcirc = (((1 / 2) * box_WH[0]) ** 2 + ((1 / 2) * box_WH[1]) ** 2) ** 0.5
    Rcirc = math.hypot((1 / 2) * box_WH[0], (1 / 2) * box_WH[1])
    cntr = [box_BL[0] + Rcirc * math.cos(delta + math.radians(box_Theta)),
                     box_BL[1] + Rcirc * math.sin(delta + math.radians(box_Theta))]

    return Rcirc, cntr

def draw_environment_wo_txt(ws_pos, ws_wh, rob_pos, obs_pos, tar_pos):
    fs = 10
    new_fig = plt.figure(figsize=(fs, fs))

    ax = plt.gca()
    # change default range so that new disks will work
    plt.axis('equal')
    ax.set_xlim((-1, 1))
    ax.set_ylim((-0.6, 0.6))
    rc, bc = calc_circumscribed_circle()

    ws = plt.Rectangle(ws_pos, ws_wh[0], ws_wh[1], color='k', fill=False)
    rob = plt.Circle(rob_pos, rr, color='gray')
    obs1 = plt.Circle(obs_pos[0], ro, color='red')
    obs2 = plt.Rectangle(obs_pos[1], box_WH[0], box_WH[1], color='red', angle = box_Theta)
    circ = plt.Circle(bc, rc, color='yellow', fill=False)
    tar = plt.Circle(tar_pos, rt, color='green')

    ax.add_patch(ws)
    ax.add_patch(rob)
    ax.add_patch(obs1)
    ax.add_patch(obs2)
    ax.add_patch(circ)
    ax.add_patch(tar)
    plt.show()
    return [rc, bc]

def Check_NonprehensileNeeded(radius):
    rG = 0.175 #Gripper size
    diam = 2 * radius

    if diam > rG:
        return True




def main():
    obs_pos = [-0.1, -0.2]
    tar_pos = [0, 0]
    R_pose = [0, -0.5]  # robot position
    ws_BL = [-0.5, -0.4]   #workspace bottom-left coordinate
    ws_size = [1.0, 0.8]  #workspace size

    ox = []
    oy = []

    rect = draw_environment_wo_txt(ws_BL, ws_size, R_pose, [obs_pos, box_BL], tar_pos)
    object = [obs_pos, rect[1], tar_pos]

    radius = [ro, rect[0], rt, rr]
    path, path_pos, Flag, goal_ori, goal_off = TP.relocate_planner(object, object, R_pose, radius)

    sx = object[path][0]
    sy = object[path][1]

    Nprehensile = Check_NonprehensileNeeded(radius[path])
    if Nprehensile:
        print("Push") #push(object_all[path])
    else:
        print("Pick and Place") #pick_and_place(object_all[path])


    for i in myrange(ws_BL[1], ws_BL[1] + ws_size[1], 0.01):
        ox.append(ws_BL[0])
        oy.append(i)
        ox.append(ws_BL[0] + ws_size[0])
        oy.append(i)
    for i in myrange(ws_BL[0], ws_BL[0] + ws_size[0], 0.01):
        ox.append(i)
        oy.append(ws_BL[1] + ws_size[1])
        ox.append(i)
        oy.append(ws_BL[1])

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r",  markersize = 15)
        plt.grid(True)
        plt.axis("equal")

    rx, ry = PRM_planning(object, radius, path, ox, oy)

    assert rx, 'Cannot found path'

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()

