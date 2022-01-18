#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of some helpfull functions to work with matplotlib."""
__author__ = "Max Schmit"
__copyright__ = "Copyright 2021, Max Schmit"

import matplotlib as mpl

def isolate_artist(ax, artist_n):
    """Isolate one Artist in the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes in which the Artist should get isolated artist
    artist_n : int
        The index of the artist. must be smaller than the number of children in ax.

    Returns
    -------
    matplotlib.figure.Figure
        The plot figure, where the element is now isolated.

    Raises
    ------
    ValueError
        If the value_n is too high. higher than the amount of children in the ax.
    """
    children = ax.get_children()
    if artist_n >= len(children):
        raise ValueError("The artist_n was higher than there were elements in the ax.")

    # set the blocking patch to cover the background
    if ("max_block_patch" not in dir(ax)):
        block_patch = mpl.patches.Rectangle((ax.get_xlim()[0], ax.get_ylim()[0]), 
                                   width=ax.get_xlim()[1] - ax.get_xlim()[0],
                                   height=ax.get_ylim()[1] - ax.get_ylim()[0],
                                   color="#D2D2D2CC",
                                   zorder=9998
                                  )
        ax.max_block_patch = block_patch
        ax.add_patch(block_patch)

    art = children[artist_n]
    for art_i in children:
        if art_i == art:
            art_i.old_zorder = art_i.get_zorder()
            art_i.set_zorder(9999)
        else:
            if art_i.get_zorder() == 9999:
                if "old_zorder" in dir(art): 
                    art_i.set_zorder(art_i.old_zorder)
                else:
                    art_i.set_zorder(1)
    return ax.get_figure()