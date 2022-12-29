import os.path

import yaml


def get_root_path():
    cur_path = os.path.dirname(__file__)
    return cur_path[:cur_path.rfind('\\')]


def get_package_path(package_name):
    return get_root_path() + '/' + package_name


def get_yml_path():
    return get_root_path() + "/" + 'data.yml'


def yml_read():
    path = get_yml_path()
    yml_data = open(path).read()
    yml_reader = yaml.load(yml_data, Loader=yaml.FullLoader)
    return yml_reader
