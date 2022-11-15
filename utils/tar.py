"""This file has 2 functions to fix a security bug in the python tarfile code. (CVE-2007-4559)

The solution was proposed by TrellixVulnTeam from the Advanced Research Center at Trellix.
"""
import os

def is_within_directory(directory, target):
    """Checks whether the filepath lies inside the directory.

    Parameters
    ----------
    directory : str or pathlike object
        The directory where to store the extracted files to
    target : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """   
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    """Make a safe extract of a tarfile.

    Parameters
    ----------
    tar : tarfile.TarFile 
        The tarfile object you want to extract.
    path : str, optional
        The directory where to store the results. 
        The default is ".".
    members : list of strings, optional
        The members to extract.
        If None, then all are extracted.
        The default is None.
    numeric_owner : bool, optional
        If numeric_owner is True, the uid and gid numbers from the tarfile are used to set the owner/group for the extracted files. Otherwise, the named values from the tarfile are used.
        The default is False

    Raises
    ------
    Exception
        If tarfile was not sanitized and travels back the directory
    """
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner) 