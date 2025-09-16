"""Script which demonstrates multiple constructors with overloading using metaclasses, based on number of positional arguments or presence of specific keywords.

- # FROM: https://github.com/dabeaz/python-cookbook/blob/master/src/9/multiple_dispatch_with_function_annotations/example1.py
- # FROM: https://stackoverflow.com/questions/141545/how-to-overload-init-method-based-on-argument-type
- # FROM: https://www.youtube.com/watch?v=yWzMiaqnpkI

"""

import inspect, types

class OverloadMethod:
    """Represents a single overloaded method.
    
    # FROM: https://github.com/dabeaz/python-cookbook/blob/master/src/9/multiple_dispatch_with_function_annotations/example1.py
     
    """
    def __init__(self, name, class_name=None):
        self._methods = {}
        self.__name__ = name
        self._class_name = class_name or ""

    def register(self, meth):
        sig = inspect.signature(meth)
        self._methods[meth] = [name for name in sig.parameters if name != 'self']

    def __call__(self, *args, **kwargs):
        """Dispatch based on number of positional args, or specific keyword selection.
        
        # FROM: GitHub Copilot, GPT-4.1

        """
        instance = args[0] if args and hasattr(args[0], '__class__') else None
        args_ = args[1:] if instance is not None else args

        if args_:
            valid_counts = []
            for meth in self._methods.keys():
                sig = inspect.signature(meth)
                param_names = [n for n in sig.parameters if n != 'self']
                params = [p for n, p in sig.parameters.items() if n != 'self']
                required_positional = [
                    p for p in params
                    if p.default is inspect.Parameter.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                all_positional = [
                    p for p in params
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
                valid_counts.append(len(all_positional))
                # Only call if the number of args is valid
                if len(args_) >= len(required_positional) and len(args_) <= len(all_positional):
                    return meth(instance, *args_, **kwargs) if instance is not None else meth(*args_, **kwargs)
            valid_counts = sorted(set(valid_counts))
            valid_counts_str = " or ".join(str(c) for c in valid_counts if c > 0)
            raise TypeError(f"{self._class_name}.{self.__name__}() takes either {valid_counts_str} positional arguments but {len(args_)} were given") from None
        else:
            for meth, param_names in self._methods.items():
                sig = inspect.signature(meth)
                required_params = [
                    n for n, p in sig.parameters.items()
                    if n != 'self' and p.default is inspect.Parameter.empty
                ]
                if all(k in kwargs for k in required_params):
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
                    return meth(instance, **filtered_kwargs) if instance is not None else meth(**filtered_kwargs)
            raise TypeError("No matching method for given keyword arguments.")
    
    def __get__(self, instance, cls):
        if instance is not None:
            return types.MethodType(self, instance)
        else:
            return self
    
class MultiDict(dict):
    """Special dictionary to build overloaded methods in a metaclass."""
    def __init__(self, *args, **kwargs):
        self._class_name = None
        super().__init__(*args, **kwargs)

    def set_class_name(self, class_name):
        self._class_name = class_name

    def __setitem__(self, key, value):
        if hasattr(value, '__overload__'):
            if key in self:
                current_value = self[key]
                if isinstance(current_value, OverloadMethod):
                    current_value.register(value)
                else:
                    mvalue = OverloadMethod(key, self._class_name)
                    mvalue.register(current_value)
                    mvalue.register(value)
                    super().__setitem__(key, mvalue)
            else:
                mvalue = OverloadMethod(key, self._class_name)
                mvalue.register(value)
                super().__setitem__(key, mvalue)
        else:
            super().__setitem__(key, value)

class OverloadBasedOnVariableNameMeta(type):
    def __new__(cls, clsname, bases, clsdict):
        if hasattr(clsdict, 'set_class_name'):
            clsdict.set_class_name(clsname)
        return type.__new__(cls, clsname, bases, dict(clsdict))

    @classmethod
    def __prepare__(cls, clsname, bases):
        d = MultiDict()
        d.set_class_name(clsname)
        return d

def overload(f):
    f.__overload__ = True
    return f

# ------ TWO CLASSES ------

class Bar:
    pass

class Foo(metaclass=OverloadBasedOnVariableNameMeta):
    """This is a dummy class with multiple constructors."""

    @overload
    def __init__(self, A: Bar, b: Bar, *, arg_1: str = 'default', arg_2: float = None):
        """Constructor with A, b, arg_1, arg_2"""
        print("Constructor with A, b, arg_1, arg_2")
        self.A = A
        self.b = b
        self.V = None
        self.arg_1 = arg_1
        self.arg_2 = arg_2

    @overload
    def __init__(self, V: Bar, *, arg_1: str = 'default', arg_2: float = None):
        """Constructor with V, arg_1, arg_2"""
        print("Constructor with V, arg_1, arg_2")
        self.A = None
        self.b = None
        self.V = V
        self.arg_1 = arg_1
        self.arg_2 = arg_2

    def baz(self):
        pass

    def baz(self, x):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} " + "{0!r}>".format(self.__dict__)
    
def foo(*args, **kwargs) -> Foo:
    """This provides a better docstring, where we can explain the multiple constructors."""
    return Foo(*args, **kwargs)

A, b, V = float(), float(), float()

# d = foo()
d1 = Foo(A, b)
print(d1.__repr__())
d2 = Foo(V, arg_1='custom', arg_2=2.71)
print(d2.__repr__())
d3 = Foo(A=A, b=b, arg_2=3.14)
print(d3.__repr__())
d4 = Foo(V=V)
print(d4.__repr__())
# d5 = Foo(A, b, V)
d6 = Foo(A, b=b, V=V)