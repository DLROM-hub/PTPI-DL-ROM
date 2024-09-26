################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Plotting utils <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import matplotlib as mpl


def im_colorbar(
    im : mpl.image,
    im_cax : mpl.image = None, 
    spacing = 0.01, 
    start_bottom = 0.0
) -> mpl.colorbar:
    """ Adds a colorbar.

    Args:
        im (mpl.image): The image to get vmin and vmax from.
        im_cax (mpl.image): The image to construct the colorbar next to
                                   (defaults to None).
        spacing (float): The spacing between im_cax and the colorbar
        start_bottom (float): The distance from the bottom ground of im_cax and
                              the bottom ground of the colorbar
    
    Returns:
        cbar (mpl.colorbar): The colorbar.                
    """

    if im_cax == None:
        im_cax = im
    l, b, w, h = im_cax.axes.get_position().bounds 
    cax = im_cax.axes.figure.add_axes(
        [l + w + spacing, b + start_bottom, 0.01, h - start_bottom]
    )
    cbar = mpl.pyplot.colorbar(im, cax = cax)

    return cbar

