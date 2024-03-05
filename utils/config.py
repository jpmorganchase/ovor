# SPDX-License-Identifier: MIT
# Copyright : JP Morgan Chase & Co

from pathlib import Path
import yaml

class Config(dict):
    """
    Modified from EasyDict: https://github.com/makinacorpus/easydict
    """
    def __init__(self, d=None, yaml_path=None, **kwargs):
        d = {} if d is None else dict(d)
        if yaml_path is not None:
            d.update(**yaml.load(open(yaml_path, 'r'), Loader=yaml.Loader))
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        reserved_fns = ('update', 'pop', 'to_dict', 'dump', 'flatten', 'merge')
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in reserved_fns:
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super().__setattr__(name, value)
        super().__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)
    
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.__dict__.items()}

    def dump(self, path=None):
        if path is None:
            return yaml.dump(self.to_dict())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return yaml.dump(self.to_dict(), open(path, 'w'), default_flow_style=False)

    def flatten(self, prefix=''):
        d = dict()
        for k, v in self.__dict__.items():
            new_k = k if not prefix else prefix + '_' + k
            if isinstance(v, Config):
                d.update(**v.flatten(new_k))
            else:
                d[new_k] = v
        return d

    def merge(self, cfg):
        if isinstance(cfg, str):
            cfg = Config(yaml_path=cfg)
        elif not isinstance(cfg, Config):
            cfg = Config(d=cfg)
        for k, v in cfg.items():
            if k in self.__dict__ and isinstance(v, Config) and isinstance(getattr(self, k), Config):
                getattr(self, k).merge(v)
            else:
                self.__setattr__(k, v)