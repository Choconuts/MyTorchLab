import json, os
import argparse
import optparse


class Argument:
    """
    class TrainOption:
        arg = Argument(int, -100, 'this is an argument with help')

    if you don't need help info, you may not need use this class

    """
    def __init__(self, type, default, help='', required=''):
        self.kwargs = {
            'help': help,
            'type': type,
            'default': default,
            'required': required,
        }


def easy_parse_all_args(opt_class):
    """
    传入自定义的option类，把所有静态成员作为argument从命令行参数入，名称保持一致
    :param opt_class: 定义若干有默认值的静态成员；可以多层继承
    :param help_dict: 需要额外帮助信息的字典，key为参数名称，value为帮助信息
    :return: 类型opt class的parse结果
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--abc')
    for k, v in vars(opt_class).items():
        if not k.startswith('__'):
            if isinstance(v, Argument):
                parser.add_argument('--' + k, **v.kwargs)
            else:
                parser.add_argument('--'+k, type=type(v), default=v, help='')
    opt = opt_class()
    ps = parser.parse_args()
    for k in vars(opt_class).keys():
        if not k.startswith('__'):
            opt.__setattr__(k, getattr(ps, k))

    return opt


class OptionBase:
    def __init__(self):
        easy_parse_all_args_to_obj(self)


def easy_parse_all_args_to_obj(opt: OptionBase):
    """
    传入自定义的option类，把所有静态成员作为argument从命令行参数入，名称保持一致
    :param opt_class: 定义若干有默认值的静态成员；可以多层继承
    :param help_dict: 需要额外帮助信息的字典，key为参数名称，value为帮助信息
    :return: 类型opt class的parse结果
    """
    parser = argparse.ArgumentParser()
    opt_class = type(opt)
    parser.add_argument('--abc')
    for k, v in vars(opt_class).items():
        if not k.startswith('__'):
            if isinstance(v, Argument):
                parser.add_argument('--' + k, **v.kwargs)
            else:
                parser.add_argument('--'+k, type=type(v), default=v, help='')

    ps = parser.parse_args()
    for k in vars(opt_class).keys():
        if not k.startswith('__'):
            opt.__setattr__(k, getattr(ps, k))

    return opt


if __name__ == '__main__':
    pass
