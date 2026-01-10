"""Script to test plotting a line which is 'indefinite', same as axhline or axvline in matplotlib, where panning/zooming works as expected"""

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def plot_line_2d(ax: Axes, line: ArrayLike, point: ArrayLike = np.array([0, 0]), **kwargs) -> None:
    """Plot an indefinite line in 2D, defined by a point and a direction vector"""

    def update_line():
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        dx, dy = vx, vy
        t_values = []
        if abs(dx) > 1e-10:
            t_left = (xlim[0] - center_x) / dx
            t_right = (xlim[1] - center_x) / dx
            t_values.extend([t_left, t_right])
        if abs(dy) > 1e-10:
            t_bottom = (ylim[0] - center_y) / dy
            t_top = (ylim[1] - center_y) / dy
            t_values.extend([t_bottom, t_top])
        if abs(dx) < 1e-10:
            x_coords = [center_x, center_x]
            y_coords = [ylim[0] - abs(ylim[1] - ylim[0]), ylim[1] + abs(ylim[1] - ylim[0])]
        elif abs(dy) < 1e-10:
            x_coords = [xlim[0] - abs(xlim[1] - xlim[0]), xlim[1] + abs(xlim[1] - xlim[0])]
            y_coords = [center_y, center_y]
        else:
            t_min, t_max = min(t_values), max(t_values); t_range = t_max - t_min
            t_min -= t_range * 0.1
            t_max += t_range * 0.1
            x_coords, y_coords = [center_x + t_min * dx, center_x + t_max * dx], [center_y + t_min * dy, center_y + t_max * dy]
        line.set_xdata(x_coords), line.set_ydata(y_coords)
    
    point = np.array(point, dtype=float)
    line = np.array(line, dtype=float)
    if np.allclose(line, 0):
        raise ValueError("Vector cannot be zero")
    
    # Calculate the center point and direction from the span vector
    x0, y0 = point
    vx, vy = line
    center_x = x0 + vx / 2
    center_y = y0 + vy / 2
    
    line, = ax.plot([], [], **kwargs)
    ax.callbacks.connect('xlim_changed', lambda ax: update_line())
    ax.callbacks.connect('ylim_changed', lambda ax: update_line())
    update_line()


def plot_line_3d(ax, line: ArrayLike, point: ArrayLike = np.array([0, 0, 0]), **kwargs) -> None:
    """Plot an indefinite line in 3D, defined by a point and a direction vector"""

    def update_line():
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        dx, dy, dz = vx, vy, vz
        t_values = []
        
        if abs(dx) > 1e-10:
            t_left = (xlim[0] - center_x) / dx
            t_right = (xlim[1] - center_x) / dx
            t_values.extend([t_left, t_right])
        if abs(dy) > 1e-10:
            t_bottom = (ylim[0] - center_y) / dy
            t_top = (ylim[1] - center_y) / dy
            t_values.extend([t_bottom, t_top])
        if abs(dz) > 1e-10:
            t_near = (zlim[0] - center_z) / dz
            t_far = (zlim[1] - center_z) / dz
            t_values.extend([t_near, t_far])
        
        if len(t_values) > 0:
            t_min, t_max = min(t_values), max(t_values)
            
            # Check if line actually intersects the 3D box
            if abs(dx) > 1e-10:
                x_min_t = (xlim[0] - center_x) / dx
                x_max_t = (xlim[1] - center_x) / dx
                if x_min_t > x_max_t:
                    x_min_t, x_max_t = x_max_t, x_min_t
            else:
                # Line parallel to x-planes: check if within x bounds
                if center_x < xlim[0] or center_x > xlim[1]:
                    x_coords, y_coords, z_coords = [], [], []
                    line_obj.set_data_3d(x_coords, y_coords, z_coords)
                    return
                x_min_t, x_max_t = -1e10, 1e10
                
            if abs(dy) > 1e-10:
                y_min_t = (ylim[0] - center_y) / dy
                y_max_t = (ylim[1] - center_y) / dy
                if y_min_t > y_max_t:
                    y_min_t, y_max_t = y_max_t, y_min_t
            else:
                # Line parallel to y-planes: check if within y bounds
                if center_y < ylim[0] or center_y > ylim[1]:
                    x_coords, y_coords, z_coords = [], [], []
                    line_obj.set_data_3d(x_coords, y_coords, z_coords)
                    return
                y_min_t, y_max_t = -1e10, 1e10
                
            if abs(dz) > 1e-10:
                z_min_t = (zlim[0] - center_z) / dz
                z_max_t = (zlim[1] - center_z) / dz
                if z_min_t > z_max_t:
                    z_min_t, z_max_t = z_max_t, z_min_t
            else:
                # Line parallel to z-planes: check if within z bounds
                if center_z < zlim[0] or center_z > zlim[1]:
                    x_coords, y_coords, z_coords = [], [], []
                    line_obj.set_data_3d(x_coords, y_coords, z_coords)
                    return
                z_min_t, z_max_t = -1e10, 1e10
            
            # Find intersection with 3D box
            t_enter = max(x_min_t, y_min_t, z_min_t)
            t_exit = min(x_max_t, y_max_t, z_max_t)
            
            # Only show line if it actually intersects the box
            if t_enter <= t_exit:
                x_coords = [center_x + t_enter * dx, center_x + t_exit * dx]
                y_coords = [center_y + t_enter * dy, center_y + t_exit * dy]
                z_coords = [center_z + t_enter * dz, center_z + t_exit * dz]
            else:
                # Line doesn't intersect current view - hide it
                x_coords, y_coords, z_coords = [], [], []
        else:
            # Handle case where line is parallel to all coordinate planes
            range_max = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0]))
            t_min, t_max = -range_max * 0.1, range_max * 0.1
            x_coords = [center_x + t_min * dx, center_x + t_max * dx]
            y_coords = [center_y + t_min * dy, center_y + t_max * dy]
            z_coords = [center_z + t_min * dz, center_z + t_max * dz]
        
        line_obj.set_data_3d(x_coords, y_coords, z_coords)
    
    point = np.array(point, dtype=float)
    line = np.array(line, dtype=float)
    if np.allclose(line, 0):
        raise ValueError("Vector cannot be zero")
    
    # Calculate the center point and direction from the span vector
    x0, y0, z0 = point
    vx, vy, vz = line
    center_x = x0 + vx / 2
    center_y = y0 + vy / 2
    center_z = z0 + vz / 2
    
    line_obj, = ax.plot([], [], [], **kwargs)
    ax.callbacks.connect('xlim_changed', lambda _: update_line())
    ax.callbacks.connect('ylim_changed', lambda _: update_line())
    ax.callbacks.connect('zlim_changed', lambda _: update_line())
    update_line()


if __name__ == "__main__":

    # ------ 2D LINE ------
    fig, ax = plt.subplots()
    
    plot_line_2d(ax, line=[1, 1], color='red', linewidth=2)
    plot_line_2d(ax, point=[1, 1], line=[-1, 2], color='blue')
    ax.plot([0, 2, 1, 0], [0, 1, 1, 3], 'go', label='Points')

    # ------ 3D LINE ------
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plot_line_3d(ax, line=[1, 1, 0], color='red', linewidth=2)
    plot_line_3d(ax, line=[-1, 2, 1], color='blue')
    ax.scatter([0, 2, 1, 0], [0, 1, 1, 3], [1, 0, 2, 3], c='g', label='Points', axlim_clip=True)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    # Show plots
    plt.show()