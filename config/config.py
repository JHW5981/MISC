"""
文件名：config.py
作者：季华伟
创建日期：2024/2/22
目的：实现detectron2的congig功能

说明：原版config.py的copy，学习一下人家的代码规范和编程技巧
"""

import functools
import inspect
import logging

from fvcore.common.config import CfgNode as _CfgNode

from ..utils.file_io import PathManager

class CfgNode(_CfgNode):
   """
   The same as `fvcore.common.config.CfgNode`, but different in:
   1. Use unsafe yaml loading by default.
      Note that this may lead to arbitrary code execution: you must not 
      load a config file from untrusted sources before manually inspecting
      the content of the file.
   2. Support config versioning.
      When attempting to merge an old config, it will convert the old config automatically.
      
   .. automethod:: clone NOTE: 用来生成自动文档
   .. automethod:: freeze
   .. automethod:: defrost
   .. automethod:: is_frozen
   .. automethod:: load_yaml_with_base
   .. automethod:: merge_from_list
   .. automethod:: merge_from_other_cfg
   """
   
   @classmethod
   def _open_cfg(cls, filename): # NOTE: "_"表示是类内部的私有函数，不供用户调用
       return PathManager.open(filename, "r")

   # Note that the default value of allow_unsafe is changed to True
   def merge_from_file(self, cfg_filename: str, allow_unsafe: bool=True) -> None:
       """
       Load content from the given config file and merge it into self.

       Args:
            cfg_filename: config filename
            allow_unsafe: all unsafe yaml syntax
       """
       assert PathManager.isfile(cfg_filename), f"Config file {cfg_filename} does not exist!"
       loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
       loaded_cfg = type(self)(loaded_cfg)
      
       from .defaults import _C
       latest_ver = _C.VERSION
       assert (
           latest_ver == self.VERSION
       ), "CfgNode.merge_from_file is only allowed on a config object of latest version."
       logger = logging.getLogger(__name__)
       loaded_ver  = loaded_cfg.get("VERSION", None)
       if loaded_ver is None:
           from .compat import guess_version
           
           loaded_ver = guess_version(loaded_cfg, cfg_filename)
       assert loaded_ver <= self.VERSION, "Cannot merge a v{} config into a v{} config".format(loaded_ver, self.VERSION)
       if loaded_ver == self.VERSION:
           self.merge_from_other_cfg(loaded_cfg)
       else:
           # compat.py needs to import CfgNode
           from .compat import upgrad_config, downgrade_config

           logger.warning(
               "Loading an old v{} config file '{}' by automatically upgrading to v{}."
               "See docs/CHANGELOG.md for instructions to update your files.".format(
                   loaded_ver, cfg_filename, self.VERSION
               )
           )
            # To convert, first obtain a full config at an old verision
           old_self = downgrade_config(self, to_version=loaded_ver)
           old_self.merge_from_other_cfg(loaded_cfg)
           new_config = upgrad_config(old_self)
           self.clear()
           self.update(new_config)
   
   def dump(self, *args, **kwargs):
      """
      Returns:
         str: a yaml string representation of the config
      """
      # to make it show up in docs
      return super().dump(*args, **kwargs)
   
global_cfg = CfgNode()

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
      a detectron2 CfgNode instance
    """
    from .defaults import _C
   
    return _C.clone()


def set_gloabal_cfg(cfg: CfgNode) -> None:
    """
    Let the global config point to the given cfg.
    Assume that the given "cfg" has the key "KEY", after calling
    `set_global_cfg(cfg)`, the key can be accessed by:
    ::    NOTE:指示下面是个代码块
         from detectron2.config import global_cfg
         print(global_cfg.KEY)

    By using a hacky global config, you can access these configs anywhere,
    without having to pass the config object or the values deep into the code.
    This is a hacky feature introduced for quick prototyping 
    """
    global global_cfg
    global_cfg.clear()
    global_cfg.update(cfg)

def configurable(init_func=None, *, from_config=None): # NOTE: *在此处的作用用于分割位置参数和关键字参数，*之后参数必须关键字输入
    """
    Decorate a function or a class's __init__ method so that it can be called 
    with a :class:`CfgNode` object using a :func:`from_config` function that translates
    :class:`CfgNode` to argments. NOTE: :name:表示name是一个标签

    Examples:
    ::
        # Usage 1: Decorator on __init__:
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass
            
            @classmethod
            def from_config(cls, cfg): # 'cfg' must be the first argument
                # Returns kwargs to be passed to __init__
                return {"a": cfg.A, "b": cfg.B}
        
        a1 = A(a=1, b=2) # regular construction
        a2 = A(cfg) # construct with a cfg
        a3 = A(cfg, b=3, c=4) # construct with extra overwrite

        # Usage 2: Decorator on any function. Needs an extra from_config argument:
        @configurable(from_config=lambda cfg:{"a": cfg.A, "b": cfg.B})
        def a_func(a, b=2, c=3):
            pass
        
        a1 = a_func(a=1,  b=2) # regular call
        a2 = a_func(cfg) # call with a cfg
        a3 = a_fucnc(cfg, b=3, c=4) # call with extra overwrite
    
        
    Args:
        init_func (callable): a class's ``__init__`` method in usage 1. The class must have 
        a ``from_config`` classmethod which takes `cfg` as the first argument.
        from_config (callable)： the from_config function in usage 2. It must take `cfg` 
        as it first argument.
    """

    if init_func is not None:
        assert (
            inspect.isfunction(init_func)
            and from_config is None
            and init_func.__name__ == "__init__"
        ), "Incorrect use of @configurable. Check API documentation for examples."

        @functools.wraps(init_func) # NOTE:保证init_func函数性质不变
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a `from_config` classmethod."
                ) from e # NOTE: 异常链
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a `from_config` classmethod.")
            
            if _called_with_cfg(*args, **kwargs):
                explicit_args  = _get_args_from_config(from_config_func, *args, **kwargs)
                init_func(self, **explicit_args)
            else:
                init_func(self, *args, **kwargs)
        return wrapped
    else:
        if from_config is None:
            return configurable # @configurable() is made equivalent to @configurable
        assert inspect.isfunction(from_config), "from_config argument of configurable must be a function!"

        def warpper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args,**kwargs):
                if _called_with_cfg(*args,**kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)
            wrapped.from_config = from_config
            return wrapped
        return warpper
    

def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.
    
    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        if inspect.isfunction(from_config_func):
            name = from_config_func.__name__
        else:
            name = f"{from_config_func.__self__}.from_config"
        raise TypeError(f"{name} must take `cfg` as the first argument!")
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg: # forward all arguments to from config , if from_config accecpts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to form_config
        support_args_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in support_args_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret

def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    from omegaconf import DictConfig

    if len(args) and isinstance(args[0], (_CfgNode, DictConfig)):
        return True
    if isinstance(kwargs.pop('cfg', None), (_CfgNode, DictConfig)):
        return True
    
    # `from_config`'s first argument is forced to be "cfg"
    # so the above check cover all cases
    return False
    



