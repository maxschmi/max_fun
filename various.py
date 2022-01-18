#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of some random helpfull functions."""

__author__ = "Max Schmit"
__copyright__ = "Copyright 2021, Max Schmit"

import sys

def clear_all_variables(globals):
    """Delete all the loaded variables.

    Parameters
    ----------
    globals : dict
        The global scope to be cleared.
        just use globals() as input.
    """
    var_names = list(globals.keys())
    for name in var_names:
        if not name.startswith('_'):
            del globals[name]

def get_objects_size(globals, min_kb=1000):
    """Get the memory size of the loaded objects.

    Parameters
    ----------
    globals : dict
        The global or local scope use globals()
    min_kb : int, optional
        The minimal memory size in kB of an object to print.
        To filter the small objects.
        The default is 1000.
    """ 
    obj_sizes = []
    obj_names = []
    for ob in globals.keys():
        size = sys.getsizeof(globals[ob])/1000
        if size >= min_kb:
            obj_names.append(ob)
            obj_sizes.append(size)

    # sort descending
    obj_sizes_unsorted = obj_sizes.copy()
    obj_sizes.sort(reverse=True)
    obj_names = [obj_names[obj_sizes_unsorted.index(obj_sizes[i])] 
                    for i in range(len(obj_sizes))]
    for name, size in zip(obj_names, obj_sizes):
        print("{name}: {size:,.0f} kB".format(name=name, size=size))