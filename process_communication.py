#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A collection of functions for processes to communicate between each other."""
from pathlib import Path
import time
import json

this_dir = Path(__file__).parent.absolute()
COM_FILE = this_dir.joinpath("data", 'process_communication.json')

def _check_com_file():
    if not COM_FILE.is_file():
        json.dump({}, open(COM_FILE, 'w'))

def send_message(key, message):
    """Send a message to another process.

    Parameters
    ----------
    key : str
        The key of the message.
    message : str
        The message to sent.
    """    
    # set finished to communicating dictionary
    _check_com_file()

    com_dict = json.load(fp=COM_FILE.open("r"))
    com_dict.update({key: message})
    json.dump(com_dict, fp=COM_FILE.open("w"))

def await_message(key, message, min_time=0, interval_time=5):
    """Await until another process sends a message.

    Parameters
    ----------
    key : str
        The key of the message.
    message : str
        The message of the variable.
    min_time : int, optional
        The minimum time in seconds to wait until message checking starts
        The default is 0.
    interval_time : int, optional
        The interval time in seconds to check for a new message
        The default is 5
    """    
    # wait until other process finishes
    _check_com_file()
    time.sleep(min_time)
    while True:
        com_dict = json.load(fp=COM_FILE.open("r"))
        if key in com_dict and com_dict[key] == message:
            print(f"start now at {time.ctime()}")
            return
        else:
            time.sleep(interval_time)