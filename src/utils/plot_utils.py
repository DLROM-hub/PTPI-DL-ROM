################################################################################
# PTPI-DL-ROMs: pre-trained physics-informed deep learning-based reduced order 
# models for nonlinear parametrized PDEs in small data regimes
#
# -> Plotting utils <-
#
# Authors:     Simone Brivio, Stefania Fresca, Andrea Manzoni
# Affiliation: MOX Laboratory (Department of Mathematics, Politecnico di Milano)
################################################################################

import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.colorbar as colorbar





def im_colorbar(
    im : img.AxesImage,
    im_cax : img.AxesImage = None, 
    spacing = 0.01, 
    start_bottom = 0.0
) -> colorbar.Colorbar:
    """ Adds a colorbar.

    Args:
        im (img.AxesImage): The image to get vmin and vmax from.
        im_cax (img.AxesImage): The image to construct the colorbar next to
                                   (defaults to None).
        spacing (float): The spacing between im_cax and the colorbar
        start_bottom (float): The distance from the bottom ground of im_cax and
                              the bottom ground of the colorbar
    
    Returns:
        cbar (colorbar.Colorbar): The colorbar.                
    """

    if im_cax == None:
        im_cax = im
    l, b, w, h = im_cax.axes.get_position().bounds 
    cax = im_cax.axes.figure.add_axes(
        [l + w + spacing, b + start_bottom, 0.01, h - start_bottom]
    )
    cbar = plt.colorbar(im, cax = cax)

    return cbar

