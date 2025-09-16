"""A class with multiple constructors, manually dispatching based on arguments."""

class Bar:
    """A dummy class for demonstration."""
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"Bar({self.value})"


class Foo:
    def __init__(self, *args, **kwargs):
        self._A = None
        self._b = None
        self._V = None
        self.extra_args = {}
        
        # Dispatch logic to handle different constructor patterns
        if len(args) == 2:
            # Pattern: Foo(A, b, ...)
            A, b = args
            if any(k in kwargs for k in ['A', 'b', 'V']):
                raise TypeError("Cannot mix positional and keyword arguments for A, b, or V.")
            # FIXME: Maybe we should just make this a class method? No, this actually does NOT work! Because this requires conversion...
            self._init_from_ab(A, b)
        elif len(args) == 1:
            # Pattern: Foo(V, ...)
            V = args[0]
            if any(k in kwargs for k in ['A', 'b', 'V']):
                raise TypeError("Cannot mix positional and keyword arguments for A, b, or V.")
            self._init_from_v(V)
        elif len(args) == 0:
            # Pattern: Foo(A=A, b=b, ...) or Foo(V=V, ...)
            if 'A' in kwargs and 'b' in kwargs:
                if 'V' in kwargs:
                    raise TypeError("Cannot provide both (A, b) and V.")
                A = kwargs.pop('A')
                b = kwargs.pop('b')
                self._init_from_ab(A, b)
            elif 'V' in kwargs:
                V = kwargs.pop('V')
                self._init_from_v(V)
            else:
                raise TypeError("Invalid arguments. Use Foo(A, b), Foo(V), Foo(A=A, b=b), or Foo(V=V).")
        else:
            raise TypeError("Invalid number of positional arguments.")
        
        # Any remaining keyword arguments are stored here
        self.extra_args = kwargs

    @classmethod
    def from_ab(cls, A, b, **kwargs):
        """Alternative constructor from A and b."""
        instance = cls(A, b, **kwargs)
        return instance
        
    def _init_from_ab(self, A, b):
        """Initializes the object from A and b."""
        print("Initializing from A and b...")
        self._A = A
        self._b = b

    def _init_from_v(self, V):
        """Initializes the object from V."""
        print("Initializing from V...")
        self._V = V

    @property
    def A(self):
        """Lazy getter for A."""
        if self._A is None and self._V is not None:
            print("Converting V to A...")
            self._A = Bar(self._V.value + 1)
        return self._A

    @property
    def b(self):
        """Lazy getter for b."""
        if self._b is None and self._V is not None:
            print("Converting V to b...")
            self._b = Bar(self._V.value - 1)
        return self._b

    @property
    def V(self):
        """Lazy getter for V."""
        if self._V is None and self._A is not None and self._b is not None:
            print("Converting (A, b) to V...")
            self._V = Bar(self._A.value + self._b.value)
        return self._V
    

# Example usage
A, b, V = Bar(0), Bar(1), Bar(True)

# All valid constructor calls now work and dispatch to the correct helper
foo_ab_pos = Foo(A, b)
foo_v_pos = Foo(V, extra='data')
foo_ab_kw = Foo(A=A, b=b, extra='data')
foo_v_kw = Foo(V=V)