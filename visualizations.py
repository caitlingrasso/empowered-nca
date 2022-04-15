import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FFMpegWriter

import copy

import constants

def display_grid(grid, title='', show=True, colormap=constants.CMAP_CELLS, save=False, fn=''):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=16)
    ax.matshow(grid, cmap=colormap)
    if show:
        plt.show()
    elif save:
        plt.savefig(fn)
        plt.close()

def display_body_signal(grids, target=None, timestep=None, title='', cmap_signal=constants.CMAP_SIGNAL, \
                        outline_color='m', original_size=constants.GRID_SIZE, save=False, fn='', \
                        show=True, target_outline_color=constants.TARGET_OUTLINE_COLOR,\
                            colorbar=False):
    hr_body = convert_to_high_res(grids[:,:,0], original_size=original_size)
    body_outline = get_outline(hr_body)
    hr_signal = convert_to_high_res(grids[:,:,1], original_size=original_size)

    if target is not None:
        hr_target = convert_to_high_res(target, original_size=original_size)
        target_outline = get_outline(hr_target)

    fig, ax = plt.subplots(1,1)
    ax.set_title(title)
    ax.axis("off")
    psm = ax.matshow(hr_signal, cmap=cmap_signal, vmin=0, vmax=255)
    cmap = ListedColormap([[1, 1, 1, 0], outline_color])
    ax.matshow(body_outline, cmap=cmap)
    target_cmap = ListedColormap([[1, 1, 1, 0], target_outline_color])

    if target is not None:
        ax.matshow(target_outline, cmap=target_cmap)
    
    if timestep is not None:
        ax.text(3, hr_body.shape[1]-3, 'n: {}'.format(timestep), color='white', backgroundcolor='k', fontsize=17)

    if colorbar:
        cbar = fig.colorbar(psm,ax=ax, aspect=10)
        cbar.ax.tick_params(labelsize=15)

    if save:
        plt.savefig(fn, bbox_inches='tight',dpi=500, pad_inches = 0)
    if show:
        plt.show()
    plt.close()

def display_empowerment_body(empowerment, body, title='', cmap_empowerment=constants.CMAP_EMPOWERMENT, \
                                outline_color='m', original_size=constants.GRID_SIZE, save=False, fn='', \
                                    show=True, colorbar=None):
    hr_body = convert_to_high_res(body, original_size=original_size)
    body_outline = get_outline(hr_body)
    hr_empowerment = convert_to_high_res(empowerment, original_size=original_size)

    fig, ax = plt.subplots(1,1)
    ax.set_title(title)
    ax.axis("off")
    psm = ax.matshow(hr_empowerment, cmap=cmap_empowerment, vmin=0, vmax=7)
    cmap = ListedColormap([[1, 1, 1, 0], outline_color])
    ax.matshow(body_outline, cmap=cmap)

    if colorbar:
        cbar = fig.colorbar(psm,ax=ax, aspect=10)
        cbar.ax.tick_params(labelsize=15)


    if save:
        plt.savefig(fn, bbox_inches='tight',dpi=500, pad_inches = 0)
    if show:
        plt.show()

    plt.close()


def get_neighbors(a):
    b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
    neigh = np.concatenate((b[2:, 1:-1, None], b[:-2, 1:-1, None],
                            b[1:-1, 2:, None], b[1:-1, :-2, None]), axis=2)
    return neigh

def convert_to_high_res(grid, original_size, res_increase=3):
    hr_grid = np.zeros((original_size * res_increase, original_size * res_increase))
    inds = np.indices(grid.shape)
    for r in inds[0][grid > 0]:
        for c in inds[1][grid > 0]:
            hr_grid[r * res_increase:r * res_increase + res_increase,
            c * res_increase:c * res_increase + res_increase] = grid[r, c]
    return hr_grid

def get_outline(hr_grid):
    nbors = get_neighbors(hr_grid)
    for r in range(hr_grid.shape[0]):
        for c in range(hr_grid.shape[1]):
            if sum(nbors[r, c, :]) == 4:
                hr_grid[r, c] = 0
    return hr_grid

def save_movie(history, fname, title='', cmap_signal=constants.CMAP_SIGNAL, outline_color='r', original_size=constants.GRID_SIZE):

    moviewriter = FFMpegWriter()
    fig, ax = plt.subplots()
    ax.axis("off")
    moviewriter.setup(fig, fname, dpi=100)

    for t in range(len(history)):
        hr_body = convert_to_high_res(history[t][:,:,0], original_size=original_size)
        body_outline = get_outline(hr_body)
        hr_signal = convert_to_high_res(history[t][:,:,1], original_size=original_size)

        ax.matshow(hr_signal, cmap=cmap_signal)
        cmap = ListedColormap([[1, 1, 1, 0], outline_color])
        ax.matshow(body_outline, cmap=cmap)

        ax.text(3, hr_body.shape[1]-3, 't: {}'.format(t), color='white', backgroundcolor='k', fontsize=12)

        moviewriter.grab_frame()

    moviewriter.finish()
    plt.close(fig)

def visualize_empowerment(history, local_empowerment, fname, title='', cmap_signal=constants.CMAP_SIGNAL, outline_color='r', original_size=constants.GRID_SIZE, empowerment_cmap='Blues'):
    # Avg local empowerment of each cell at end of sim shown as static heatmap after development video
    moviewriter = FFMpegWriter()
    fig, ax = plt.subplots()
    ax.axis("off")
    moviewriter.setup(fig, fname, dpi=100)

    for t in range(len(history)):
        hr_body = convert_to_high_res(history[t][:,:,0], original_size=original_size)
        body_outline = get_outline(hr_body)
        hr_signal = convert_to_high_res(history[t][:,:,1], original_size=original_size)

        ax.matshow(hr_signal, cmap=cmap_signal, vmin=0, vmax=255)
        cmap = ListedColormap([[1, 1, 1, 0], outline_color])
        ax.matshow(body_outline, cmap=cmap)

        ax.text(3, hr_body.shape[1]-3, 't: {}'.format(t), color='white', backgroundcolor='k', fontsize=12)

        moviewriter.grab_frame()

    # Display empowerment heatmap as last frame
    hr_local_empowerment = convert_to_high_res(local_empowerment, original_size=original_size)
    ax.matshow(hr_local_empowerment, cmap=empowerment_cmap, vmin=0, vmax=7)
    cmap = ListedColormap([[1, 1, 1, 0], outline_color])
    ax.matshow(body_outline, cmap=cmap)
    moviewriter.grab_frame()

    moviewriter.finish()
    plt.close(fig)

def single_grid_timelapse(series, CMAP, save_fn):

    moviewriter = FFMpegWriter()
    fig, ax = plt.subplots()
    ax.axis("off")
    moviewriter.setup(fig, save_fn, dpi=100)

    for t in range(len(series)):
        ax.matshow(series[t], cmap=CMAP)
        ax.text(3, series.shape[1]-3, 'n: {}'.format(t), color='white', backgroundcolor='k', fontsize=12)

        moviewriter.grab_frame()

    moviewriter.finish()
    plt.close(fig)