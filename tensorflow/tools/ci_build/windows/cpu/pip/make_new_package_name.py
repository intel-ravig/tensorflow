#!/usr/bin/env python3
"""
Takes a newly made wheel and makes a new wheel with a similar name
(ex: intel_numpy vs numpy)
Purpose of this is so when users `pip install intel_X` they won't
get their `X` overwritten if they then do `pip install X`

Not using any external deps for this since it's a simple task.
Assumes you're in the wheel dir with your wheel name (fuzzy searches for it)

Ex: cd dist && ../make_new_package_name.py tensorflow tensorflow-mkl
    This will make a new wheel that installs tensorflow-mkl in addition to tensorflow
    The new wheel will be called `tensorflow-mkl`
"""
import fileinput
import os
import shutil
import sys
import uuid
# NOTE: need wheel>=0.32.2
# can't use `wheel pack` command because that only wants 1 .dist-info in wheel
from wheel.wheelfile import WheelFile

from contextlib import contextmanager
from zipfile import ZipFile


@contextmanager
def temp_dir():
    """creates temporary working dir we delete after"""
    temp_name = str(uuid.uuid4())
    os.makedirs(temp_name)
    try:
        yield temp_name
    finally:
        shutil.rmtree(temp_name)


def main():
    if len(sys.argv) != 3:
        raise ValueError("Need old wheel and new wheel names.")

    old_wheel = sys.argv[1]
    new_wheel = sys.argv[2]

    # there could be many wheel versions that match the name `X`
    # making copy here because we'll be creating more files later in this dir
    wheel_files = os.listdir('.')
    # keep track of wheel name so new wheel can be named similarly
    for old_wheel_full_name in wheel_files:
        # sometimes we have files like .whl.sha256sum that aren't wheels
        if old_wheel not in old_wheel_full_name or \
                not old_wheel_full_name.endswith('.whl') or \
                not old_wheel_full_name.startswith(old_wheel):
            continue

        print("Processing ", old_wheel_full_name)
        with temp_dir() as work_dir:
            with ZipFile(old_wheel_full_name) as zipp:
                zipp.extractall(work_dir)

            # keep track of new dist-info dir so we can edit files inside it
            new_dist_info_path = ''
            # TODO: could just be os.listdir filter with os.path.isdir possibly
            for root, dirs, files in os.walk(work_dir):
                for each_dir in dirs:
                    # find the dist-info dir we want
                    # we need to copy it and change its contents
                    if '.dist-info' in each_dir:
                        new_dist_info_path = os.path.join(
                            root, each_dir.replace(old_wheel, new_wheel))
                        shutil.copytree(
                            os.path.join(root, each_dir), new_dist_info_path)
                        shutil.rmtree(os.path.join(root, each_dir), True)
                        break

            if not new_dist_info_path:
                raise ValueError("Unable to copy .dist-info dir")

            # edit the newly copied dist-info contents with the new wheel name
            for fi in os.listdir(new_dist_info_path):
                if 'METADATA' in fi:
                    # need to make metadata record new_wheel as package name
                    for line in fileinput.input(
                            os.path.join(
                                new_dist_info_path, fi), inplace=True):
                        if 'Name: ' in line:
                            line = 'Name: ' + new_wheel + '\n'
                        print(line, end='')

            # create new wheel with new dist-info
            new_wheel_name = old_wheel_full_name.replace(old_wheel, new_wheel)
            with WheelFile(new_wheel_name, 'w') as wf:
                wf.write_files(work_dir)


if __name__ == '__main__':
    main()
