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
        super().__init__()
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--conf', type=str, default=None, help='configuration json file', required=False)
        # parser.add_argument('--conf_modify', type=str, default='l',
        #                     help='save configuration(s), load configuration(l) or ignore configuration(i)',
        #                     required=False)
        # opts = parser.parse_args()
        # conf_path = opts.conf
        # mode = opts.conf_modify
        # if conf_path is None or mode in ['i', 's']:
        #     easy_parse_all_args_to_obj(self)
        # else:
        #     self.load(conf_path)
        #
        # if conf_path is not None and mode is 's':
        #     self.save(conf_path)

        parser = argparse.ArgumentParser()
        parser.add_argument('--conf', type=str, default=None, help='configuration json file', required=False)
        parser.add_argument('--conf_modify', type=str, default='l',
                            help='save configuration(s), load configuration(l) or ignore configuration(i)',
                            required=False)
        obj_parser_maker(self, parser)

        opts = parser.parse_args()
        conf_path = opts.conf
        mode = opts.conf_modify
        if conf_path is None or mode in ['i', 's']:
            obj_attr_setter(self, opts)
        else:
            self.load(conf_path)

        if conf_path is not None and mode is 's':
            self.save(conf_path)

        # print('#################### Configuration ####################')
        # for k, v in self.to_json().items():
        #     print('%20s : ' % k, str(v))
        # print('#######################################################')

    def to_json(self):
        obj = {}
        for k, v in vars(type(self)).items():
            if not k.startswith('__'):
                obj[k] = v
        return obj

    def from_json(self, obj):
        for k, v in obj.items():
            self.__setattr__(k, v)

    def save(self, json_file):
        with open(json_file, 'w') as fp:
            json.dump(self.to_json(), fp)

    def load(self, json_file):
        with open(json_file, 'r') as fp:
            self.from_json(json.load(fp))


def easy_parse_all_args_to_obj(opt: OptionBase):
    """
    传入自定义的option类，把所有静态成员作为argument从命令行参数入，名称保持一致
    :param opt: 定义若干有默认值的静态成员；可以多层继承
    :param help_dict: 需要额外帮助信息的字典，key为参数名称，value为帮助信息
    :return: 类型opt class的parse结果
    """
    parser = argparse.ArgumentParser()
    opt_class = type(opt)
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


def obj_parser_maker(opt: OptionBase, parser):
    opt_class = type(opt)
    for k, v in vars(opt_class).items():
        if not k.startswith('__'):
            if isinstance(v, Argument):
                parser.add_argument('--' + k, **v.kwargs)
            else:
                parser.add_argument('--'+k, type=type(v), default=v, help='')


def obj_attr_setter(opt: OptionBase, parse_res):
    for k in vars(type(opt)).keys():
        if not k.startswith('__'):
            opt.__setattr__(k, getattr(parse_res, k))

    return opt


if __name__ == '__main__':
    pass

