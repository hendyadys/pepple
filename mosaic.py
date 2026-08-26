import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def mosaic(vol, fig=None, title=None, size=[10, 10], vmin=None, vmax=None,
           return_mosaic=False, cbar=True, return_cbar=False, **kwargs):
    """
    Display a 3-d volume of data as a 2-d mosaic

    Parameters
    ----------
    vol: 3-d array
       The data

    fig: matplotlib figure, optional
        If this should appear in an already existing figure instance

    title: str, optional
        Title for the plot

    size: [width, height], optional

    vmin/vmax: upper and lower clip-limits on the color-map

    **kwargs: additional arguments to matplotlib.pyplot.matshow
       For example, the colormap to use, etc.

    Returns
    -------
    fig: The figure handle

    """
    if vmin is None:
        vmin = np.nanmin(vol)
    if vmax is None:
        vmax = np.nanmax(vol)

    sq = int(np.ceil(np.sqrt(len(vol))))

    # Take the first one, so that you can assess what shape the rest should be:
    im = np.hstack(vol[0:sq])
    height = im.shape[0]
    width = im.shape[1]

    # If this is a 4D thing and it has 3 as the last dimension
    if len(im.shape) > 2:
        if im.shape[2] == 3 or im.shape[2] == 4:
            mode = 'rgb'
        else:
            e_s = "This array has too many dimensions for this"
            raise ValueError(e_s)
    else:
        mode = 'standard'

    for i in range(1, sq):
        this_im = np.hstack(vol[(len(vol) / sq) * i:(len(vol) / sq) * (i + 1)])
        wid_margin = width - this_im.shape[1]
        if wid_margin:
            if mode == 'standard':
                this_im = np.hstack([this_im,
                                     np.nan * np.ones((height, wid_margin))])
            else:
                this_im = np.hstack([this_im,
                                     np.nan * np.ones((im.shape[2],
                                                       height,
                                                       wid_margin))])
        im = np.concatenate([im, this_im], 0)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        # This assumes that the figure was originally created with this
        # function:
        ax = fig.axes[0]

    if mode == 'standard':
        imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, **kwargs)
    else:
        imax = plt.imshow(np.rot90(im), interpolation='nearest')
        cbar = False
    ax.get_axes().get_xaxis().set_visible(False)
    ax.get_axes().get_yaxis().set_visible(False)
    returns = [fig]
    if cbar:
        # The colorbar will refer to the last thing plotted in this figure
        cbar = fig.colorbar(imax, ticks=[np.nanmin([0, vmin]),
                                         vmax - (vmax - vmin) / 2,
                                         np.nanmin([vmax, np.nanmax(im)])],
                            format='%1.2f')
        if return_cbar:
            returns.append(cbar)

    if title is not None:
        ax.set_title(title)
    if size is not None:
        fig.set_size_inches(size)

    if return_mosaic:
        returns.append(im)

    # If you are just returning the fig handle, unpack it:
    if len(returns) == 1:
        returns = returns[0]

    return returns