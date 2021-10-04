import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

def print_traj(traj, color=None):
    """
    Method that will plot the trajectory
    """
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.plot(ep_trajectory[:, 0], ep_trajectory[:, 1], color=color)


if __name__ == '__main__':

    all_trajs = []
    for i in range(1):
        traj = [(4.5,4.5)]
        for i in range(20):
            p = traj[-1]
            offset = (np.random.randint(-1, 2), np.random.randint(-1, 2))
            p_t1 = (np.clip(p[0] + offset[0], 0, 10), np.clip(p[1] + offset[1], 0, 10))
            traj.append(p_t1)
        all_trajs.append(traj)

    fig = plt.figure()

    grids_x = [[(i, 0), (i, 10)] for i in range(10)]
    grids_y = [[(0, i), (10, i)] for i in range(10)]

    for g in grids_x + grids_y:
        print_traj(g, color='g')

    grids_x = [[(i, 0), (i, 10)] for i in range(0, 10, 2)]
    grids_y = [[(0, i), (10, i)] for i in range(0, 10, 2)]

    for g in grids_x + grids_y:
        print_traj(g, color='b')

    for t in all_trajs:
        print_traj(t)
    t = np.asarray(t)
    plt.scatter(t[:, 0], t[:, 1])
    # plt.show()

    agg_trajs = []

    width = height = 10
    agg_width = agg_height = 2

    cells = []
    for i in range(0, width, agg_width):
        for j in range(0, height, agg_height):
            cells.append((agg_width / 2 + i, agg_height / 2 + j))

    plt.figure()
    plt.grid()
    for traj in all_trajs:

        for i in range(0, len(traj) - 1):
            p_t = traj[i]
            p_t1 = traj[i + 1]

            p_t = np.reshape(p_t, (1,2))
            p_t1 = np.reshape(p_t1, (1, 2))

            cell_t = cells[np.argmin(distance_matrix(p_t, cells))]
            cell_t1 = cells[np.argmin(distance_matrix(p_t1, cells))]

            if cell_t != cell_t1:
                if len(agg_traj) == 0:
                    agg_traj.append(cell_t)
                agg_traj.append(cell_t1)

        print_traj(agg_traj)
    cells = np.asarray(cells)
    plt.scatter(cells[:, 0], cells[:, 1])
    plt.show()