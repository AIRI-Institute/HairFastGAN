"""Complementary functions for os.path."""

import fnmatch
import os
import sys


def add_path(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def split(path):
    dir, name_ext = os.path.split(path)
    name, ext = os.path.splitext(name_ext)
    return dir, name, ext


def directory(path):
    return split(path)[0]


def name(path):
    return split(path)[1]


def ext(path):
    return split(path)[2]


def name_ext(path):
    return ''.join(split(path)[1:])


abspath = os.path.abspath
join = os.path.join


def match(dir, pat, recursive=False):
    if recursive:
        iterator = os.walk(dir)
    else:
        try:
            iterator = [next(os.walk(dir))]
        except:
            return []
    matches = []
    for root, _, file_names in iterator:
        for file_name in fnmatch.filter(file_names, pat):
            matches.append(os.path.join(root, file_name))
    return matches
