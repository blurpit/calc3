import math
from copy import copy
from enum import Enum

from scipy import linalg
from scipy.linalg import LinAlgError


def is_identifier(name:str):
    return name.isidentifier()


class Associativity(Enum):
    """
    Associativity defines the direction operators and functions are evaluated when strung
    together without parentheses. For example, division is left-to-right associative,
    so 2/3/4 = (2/3)/4 whereas exponentiation is right-to-left associative, so
    2^3^4 = 2^(3^4).
    """
    L_TO_R = 1
    R_TO_L = 2

class DefinitionType(Enum):
    """
    Each token is associated with a category to differentiate them (ex. unary minus
    vs binary minus). Each token type has a different syntax in an equation. Items
    with the same type and name cannot exist simultaneously in a context.

    These are necessary to retrieve an item from the context, for example to get binary
    minus, use ``ctx.get('-', DefinitionType.BINARY_OPERATOR)``
    """
    IDENTIFIER = 0
    BINARY_OPERATOR = 1
    UNARY_OPERATOR = 2


class Definition:
    precedence = 5
    associativity = Associativity.R_TO_L
    token_type = DefinitionType.IDENTIFIER

    def __init__(self, name, args, f_args, func, **kwargs):
        """ Base class for definitions of identifiers. """
        self.name = name
        self.args = args
        self.f_args = f_args
        self.func = func
        self.star_arg = kwargs.get('star_arg', False)
        self.display_name = kwargs.get('display_name', None)
        self.help_text = kwargs.get('help_text', None)
        self.manual_eval = kwargs.get('manual_eval', False)
        self.ctx = None

    def copy(self):
        """ Make a shallow copy of this definition object """
        return copy(self)

    def bind_context(self, ctx):
        """
        Bind this definition to a context. A context must be bound in order to use __call__ if manual_eval is true, or
        if this definition is a DeclaredFunction. This is done automatically when the definition is added to a context.

        If a definition is already bound to a context, and is added to a different context using ``ctx.add()`` or
        ``ctx.set()``, a shallow copy is made and bound to the new context. So, if a function ``f`` is bound to
        ``ctx1``, then it will still be bound to ``ctx1`` after ``ctx2.add(f)`` is used. However, ``ctx2.get('f')``
        will return an identical Definition that is bound to ``ctx2``.

        If a definition is unbound, a copy will not be made when adding it to the context. Use ``.bind_context(None)``
        to unbind.

        :param ctx: hello
        """
        self.ctx = ctx

    def check_inputs(self, n_inputs):
        if len(self.args) != n_inputs and \
                (not self.star_arg or n_inputs < len(self.args) - 1):
            raise TypeError('{} expected {} argument{}, got {}'.format(
                self.signature,
                len(self.args),
                '' if len(self.args) == 1 else 's',
                n_inputs
            ))

    def __call__(self, *inputs):
        if self.is_constant:
            return self.func
        elif self.manual_eval:
            return self.func(self.ctx, *inputs)
        else:
            return self.func(*inputs)

    def add_args_to_context(self, ctx, inputs):
        """ Define args in a context as unknown definitions. """
        if inputs is None:
            # Unknown signature; match each arg to None
            for i in range(len(self.args)):
                ctx.add(_wrap_argument_definition(self.args[i], self.f_args[i], None))
        elif self.star_arg:
            # Match each arg until the star arg, then pass the star arg as a list of the
            # remaining inputs
            for i in range(len(self.args)-1):
                ctx.add(_wrap_argument_definition(self.args[i], self.f_args[i], inputs[i]))
            last = len(self.args) - 1
            ctx.add(_wrap_argument_definition(self.args[last], False, list(inputs[last:])))
        else:
            # Match each arg to each input
            for i in range(len(self.args)):
                ctx.add(_wrap_argument_definition(self.args[i], self.f_args[i], inputs[i]))

    @property
    def signature(self):
        """ The function signature, Ex. 'foo(a, b, c)'. """
        if self.is_constant:
            return self.display_name or self.name

        return '{}({})'.format(
            self.display_name or self.name,
            ', '.join(self._args_list())
        )

    @property
    def is_constant(self):
        """ Constant functions take 0 arguments and their `func` is a value instead of a callable. """
        return len(self.args) == 0 and not callable(self.func)

    def _args_list(self):
        """ Returns a list of pretty arg names, with () added for f_args and * added for star args. """
        args = []
        for i, name in enumerate(self.args):
            if self.f_args[i]:
                name += '()'
            args.append(name)
        if self.star_arg:
            args[-1] = '*' + args[-1]
        return args

    def __str__(self):
        return self.signature

    def __repr__(self):
        return '<{} name={}, args={}, func={}>'.format(
            type(self).__name__,
            repr(self.name),
            repr(self._args_list()),
            repr(self.func),
        )

def _wrap_argument_definition(arg_name, is_f_arg, value):
    """
    Wraps a given value in a new Definition with a given argument name. This is used to associate values passed into
    a function with the argument it was passed in to. For example if "pi/2" is passed into sqrt(x), then a new
    Definition is returned with a name "x" and value "pi/2". The parser uses this to declare that an identifier
    will exist at evaluation time.

    :param arg_name: Argument name
    :param is_f_arg: Whether this argument is a function
    :param value: Value of the identifier (or None if unknown)
    :return: A Definition object with a name equal to arg_name
    """
    if is_f_arg:
        return Definition(arg_name, ('...',), (False,), value, star_arg=True)
    else:
        return Definition(arg_name, (), (), value)


class FunctionDefinition(Definition):
    def __init__(self, name, args, func, precedence=None, display_name=None,
                 latex=None, help_text=None, manual_eval=False):
        """
        Define a function

        `args` should be a list of strings which represent the name of each argument. If this function takes another
        function as an argument, add '()' to the name of that argument, for example ``args=['f()', 'x']`` takes a
        function `f` and a variable `x` as inputs.

        `func` should be a callable if the function takes 1 or more arguments, otherwise it may be either a callable or
        a direct value.

        `precedence` determines what goes inside the parentheses when the function is called implicitly. Default is 3
        (between + and *). Ex. ``sin2*pi/3+8`` is parsed as ``sin(2*pi/3)+8``. Higher number = higher precedence.

        Set `display_name` to have the expression show a different name than what was passed into `name` when the
        expression is parsed and converted back to a string. Ex. if a function has a name of 'delta' and a `display_name`
        of 'Δ', then the function would use 'delta(...)' in expressions but ``str(definition)`` will be 'Δ(...)'.

        `latex` should be a function that that takes `ctx`, `node`, and `*inputs`. `node` is the ``Function`` node in
        the syntax tree representing the function call, and `*inputs` are the inputs to the function call. It should
        return a LaTeX string. For example, for the expression ``int(sin, 0, 3)``, `node` will be the ``Function`` node
        representing the integral function call (you can get ``int``'s the definition using ``ctx.get(node.name)``),
        and ``inputs`` will be ``(Number(0), Number(3))``. Use ``inputs[i].latex(ctx)`` to convert  an input to latex.
        Passing a `latex` function is useful for special functions that should be rendered differently from the standard
        ``name(a, b, c)`` format in latex, like integrals.

        `help_text` will be shown when evaluating ``help(name)``. If none is provided, it will use the docstring
        associated with `func`. If that is also not provided, it will say "No description provided."

        If `manual_eval` is True, inputs passed into `func` will not be evaluated, and instead `func` will be passed
        `ctx`, `*inputs`. The inputs are the ``Node`` objects representing the inputs to the function. Use
        ``calc.evaluate(ctx, node)`` to evaluate them. This can be useful for functions that should not evaluate all
        inputs beforehand, such as ``if(condition, if_t, if_f)``, where if the condition is true, ``if_f`` should not be
        evaluated. NOTE: Sometimes the inputs will still be evaluated beforehand, such as when the function is passed as
        an argument to another function then called.

        :param name: Function name
        :param args: Function arguments
        :param func: Function implementation
        :param precedence: Function precedence if parentheses are absent (default 2)
        :param display_name: Name to use for str(definition)
        :param latex: Function that converts a function call expression to LaTeX
        :param help_text: Text shown on help()
        :param manual_eval: If True, evaluate inputs inside `func` rather than beforehand
        """
        # Convert passed args list into args and f_args
        f_args = []
        for i, arg in enumerate(args):
            if arg.endswith('()'):
                f_args.append(True)
                args[i] = arg[:-2]
            else:
                f_args.append(False)
            if len(args[i]) == 0:
                raise ValueError('Function argument name cannot be empty')

        # Check for star arg
        star_arg = False
        for i, arg in enumerate(args):
            if arg[0] == '*':
                if len(arg) == 1:
                    raise ValueError('Function argument name cannot be empty')
                if i != len(args) - 1:
                    raise ValueError('Star argument must be last')
                args[i] = arg[1:]
                star_arg = True

        # Check arg names
        for arg in args:
            if not is_identifier(arg):
                raise ValueError("Invalid identifier name '{}'".format(arg))

        # Check func
        # If func is None we'll assume it's supposed to be replaced later. If it doesn't get replaced, that's more
        # their problem than mine.
        if func is not None and len(args) > 0 and not callable(func):
            raise ValueError("Functions that take 1 or more arguments must be passed a callable, "
                             "not '{}'".format(func))

        if precedence is not None:
            self.precedence = precedence
        self.latex_func = latex

        super().__init__(
            name, args, f_args, func,
            star_arg=star_arg, display_name=display_name,
            manual_eval=manual_eval, help_text=help_text
        )


class VariableDefinition(Definition):
    precedence = -1

    def __init__(self, name, value, display_name=None, latex=None, help_text=None):
        """
        Define a variable

        :param name: Name of the variable
        :param value: Value of the variable
        :param display_name: Name to use for str(definition)
        :param latex: Latex command to use for the variable name
        :param help_text: Text shown on help()
        """
        super().__init__(
            name, (), (), value,
            display_name=display_name, help_text=help_text
        )
        self.latex_name = latex


class DeclaredFunction(FunctionDefinition):
    """
    DeclaredFunctions are functions created using a declaration expression. For example
    ``calc.evaluate(ctx, "f(x) = 3x^2")`` will return a DeclaredFunction. Declared functions can be called like a normal
    function (``f(4)`` will return 48) or added to a context using ``ctx.add(f)``.
    """
    def __init__(self, name, args, is_const):
        super().__init__(name, args, None)
        self._is_const = is_const and len(args) == 0

    def __call__(self, *inputs):
        """ Evaluate the function. A context must be binded to use this. """
        self.check_inputs(len(inputs))

        # Check cached value
        if self._is_const and hasattr(self, '_value'):
            return self._value

        with self.ctx.with_scope():
            if not self.name in self.ctx:
                self.ctx.add(self)
            self.add_args_to_context(self.ctx, inputs)
            result = self.func.evaluate(self.ctx)
            self._value = result
            return result

    @property
    def is_constant(self):
        return self._is_const

    def __str__(self):
        if self._is_const and hasattr(self, '_value'):
            if type(self._value) == list:
                body = ', '.join(map(str, self._value))
            else:
                body = str(self._value)
        else:
            body = str(self.func)

        if type(self.func).__name__ == 'ListNode':
            # add parentheses if the function returns a list (Ex. "f(x)=(1,2,3), 4") because the comma operator has
            # lower precedence than a declaration.
            # Todo: compare precedences instead of a direct check for a ListNode type. This should, ideally, reflect
            #       Declaration.find_expression_end().
            body = '(' + body + ')'

        return '{} = {}'.format(self.signature, body)


class BinaryOperatorDefinition(Definition):
    token_type = DefinitionType.BINARY_OPERATOR

    def __init__(self, symbol, func, precedence, associativity, latex=None, help_text=None, manual_eval=False):
        """
        Define a binary operator

        `func` should be a callable that takes 2 inputs and returns the result of the operator, where the first input
        is the left operand and the 2nd is the right operand.

        `precedence` and `associativity` are required for binary operators. Precedence defines what order operators are
        evaluated when parentheses aren't used. Higher number = higher precedence.

        `latex` should be a function that takes the following inputs: `ctx`, `node`, `left`, `right`, `parens_left`,
        `parens_right`, and `is_implicit`. `node` is the BinaryOperator node in the syntax tree, and `left` and `right`
        are the operand Nodes. `parens_right` and `parens_left` say whether the left/right operands are parenthesized
        (this happens when they are lower precedence than the parent operator, Ex. "(3+5)/2"). `is_implicit` says
        whether the operator is implicit, Ex. "2*π" vs "2π". Use ``.latex(ctx)`` to convert `left` and `right` to latex.

        :param symbol: Operator symbol (must be exactly 1 character)
        :param func: Function that implements the operator
        :param precedence: Operator precedence
        :param associativity: Operator associativity (see ``Associativity``)
        :param latex: Function that converts the operator into LaTeX
        :param help_text: Text shown on help()
        :param manual_eval: See FunctionDefinition
        """
        if len(symbol) != 1:
            raise ValueError("Invalid binary operator symbol '{}'; "
                             "symbols must be 1 character".format(symbol))
        super().__init__(
            symbol, 'ab', (False, False), func,
            help_text=help_text, manual_eval=manual_eval
        )
        self.precedence = precedence
        self.associativity = associativity
        self.latex_func = latex

    def __str__(self):
        name = self.display_name or self.name
        return 'a ' + name + ' b'

    def __repr__(self):
        return '<{} symbol={}, precedence={}, func={}>'.format(
            type(self).__name__,
            repr(self.name),
            self.precedence,
            repr(self.func)
        )


class UnaryOperatorDefinition(Definition):
    precedence = 7
    associativity = Associativity.R_TO_L
    token_type = DefinitionType.UNARY_OPERATOR

    def __init__(self, symbol, func, precedence=None, latex=None, help_text=None, manual_eval=False):
        """
        Define a unary operator.

        `func` should be a callable that takes 1 input and returns the result of the operator.

        `latex` should be a function that takes the following inputs: `ctx`, `node`, `left`, `right`, `parens_left`,
        `parens_right`, and `is_implicit`. `node` is the UnaryOperator node in the syntax tree, and `right`
        are the operand Nodes. `parens_right` and `parens_left` say whether the left/right operands are parenthesized
        (this happens when they are lower precedence than the parent operator, Ex. "(3+5)/2"). `is_implicit` says
        whether the operator is implicit, Ex. "2*π" vs "2π". Use ``.latex(ctx)`` to convert `left` and `right` to latex.

        :param symbol: Operator symbol (must be exactly 1 character)
        :param func: Function that implements the operator (should take 1 input)
        :param precedence: Operator precedence
        :param latex: Function that converts the operator into LaTeX
        :param help_text: Text shown on help()
        :param manual_eval: See FunctionDefinition
        """
        if len(symbol) != 1:
            raise ValueError("Invalid unary operator symbol '{}'; "
                             "sybmols must be 1 character".format(symbol))
        super().__init__(
            symbol, 'x', (False,), func,
            help_text=help_text, manual_eval=manual_eval
        )
        if precedence is not None:
            self.precedence = precedence
        self.latex_func = latex

    def __str__(self):
        return self.name + 'x'

    def __repr__(self):
        return '<{} symbol={}, func={}>'.format(
            type(self).__name__,
            repr(self.name),
            repr(self.func)
        )


class vector(list):
    def __init__(self, *v):
        super().__init__()
        for item in v:
            if type(item) == list: self.extend(item)
            else: self.append(item)

    def __add__(self, other):
        if isinstance(other, vector):
            if len(other) == len(self):
                v = vector()
                for a, b in zip(self, other):
                    v.append(a + b)
                return v
            raise LinAlgError('vectors must be the same dimension')
        raise TypeError("unsupported operand type(s) for +: '{}' and '{}'"
                        .format(type(self).__name__, type(other).__name__))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            v = vector()
            for x in self:
                v.append(x * other)
            return v
        raise TypeError("unsupported operand type(s) for *: '{}' and '{}'"
                        .format(type(self).__name__, type(other).__name__))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            v = vector()
            for x in self:
                v.append(x / other)
            return v
        raise TypeError("unsupported operand type(s) for /: '{}' and '{}'"
                        .format(type(self).__name__, type(other).__name__))

    def dot(self, v):
        """ Returns the dot product of this vector and v """
        if isinstance(self, vector) and isinstance(v, vector):
            if len(self) == len(v):
                return sum(self[i] * v[i] for i in range(len(self)))
            raise LinAlgError('vectors must be the same dimension')
        raise TypeError('v and w must be vectors')

    @staticmethod
    def zero(d):
        """ Returns a d-dimensional zero vector """
        v = vector()
        for _ in range(d):
            v.append(0)
        return v

    def mag2(self):
        """ Returns the magnitude squared of the vector """
        if isinstance(self, vector):
            return sum(x*x for x in self)
        raise TypeError('v must be a vector')

    def mag(self):
        """ Returns the magnitude of the vector """
        if isinstance(self, vector):
            return math.sqrt(sum(x*x for x in self))
        raise TypeError('v must be a vector')

    def norm(self):
        """ Returns a normalized copy of the vector """
        if isinstance(self, vector):
            return self / self.mag()
        raise TypeError('v must be a vector')

    def i(self, r):
        """ Returns the value in the ith row in the vector """
        return self[r]

    def copy(self):
        """ Return a shallow copy of the vector """
        v = vector()
        for x in self:
            v.append(x)
        return v

    def __round__(self, n=None):
        """ Return a rounded copy of the vector """
        v = self.copy()
        for i, x in enumerate(self):
            if hasattr(x, '__round__'):
                x = round(x, n)
                if x % 1 == 0:
                    x = int(x)
                v[i] = x
        return v

    def __str__(self):
        return '<{}>'.format(', '.join(map(str, self)))

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(repr, self)))

    def latex(self, ctx):
        vec = r' \\ '.join(
            x.latex(ctx) if hasattr(x, 'latex') else str(x)
            for x in self
        )
        return r'\begin{bmatrix} ' + vec + r' \end{bmatrix}'

class matrix(list):
    def __init__(self, *rows):
        for row in rows:
            if not isinstance(row, list):
                raise TypeError("matrix row must be a list or vector, not '{}'".format(type(row).__name__))
            if len(row) != len(rows[0]):
                raise ValueError('matrix must be rectangular')

        self.shape = [
            len(rows),
            len(rows[0]) if len(rows) > 0 else 0
        ]

        super().__init__()
        for row in rows:
            self.append(vector(*row))

    def __mul__(self, other):
        # matrix-scalar multiplication
        if isinstance(other, (int, float)):
            return matrix(*(
                other * row
                for row in self
            ))

        # matrix-matrix multiplication
        elif isinstance(other, matrix):
            if self.shape[1] == other.shape[0]:
                m = matrix.zero(self.shape[0], other.shape[1])
                for r in range(self.shape[0]):
                    for c in range(other.shape[1]):
                        for k in range(self.shape[1]):
                            m[r][c] += self[r][k] * other[k][c]
                return m
            raise LinAlgError('incompatible shapes for matrix multiplication')

        # matrix-vector multiplication
        elif isinstance(other, vector):
            if len(other) == self.shape[1]:
                v = vector.zero(self.shape[0])
                for r in range(self.shape[0]):
                    for c in range(self.shape[1]):
                        v[r] += other[c] * self[r][c]
                return v
            raise LinAlgError('incompatible shapes for matrix-vector multiplication')

        raise TypeError("unsupported operand type(s) for *: '{}' and '{}'"
                        .format(type(self).__name__, type(other).__name__))

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        raise TypeError("unsupported operand type(s) for *: '{}' and '{}'"
                        .format(type(other).__name__, type(self).__name__))

    def __pow__(self, power, modulo=None):
        if isinstance(power, int):
            if self.shape[0] == self.shape[1]:
                if power == -1:
                    # M^-1 = inverse of M
                    arr = linalg.inv(self)
                    return matrix.from_numpy(arr)
                elif power == 0:
                    # M^0 = identity matrix
                    return matrix.id(self.shape[0])
                elif power > 0:
                    # M^n = M*M*M*...
                    m = self
                    for _ in range(power-1):
                        m = m * self
                    return m
                raise LinAlgError('matrix power must be -1 or greater')
            raise LinAlgError('matrix must be square')
        raise TypeError("unsupported operand type(s) for ** or pow(): '{}' and '{}'"
                        .format(type(self).__name__, type(power).__name__))

    def __neg__(self):
        return self * -1

    def transp(self):
        """ Returns the transpose of the matrix """
        return matrix(*(self.col(c) for c in range(self.shape[1])))

    def row(self, r):
        """ Returns the rth row vector of the matrix """
        return self[r]

    def col(self, c):
        """ Returns the cth column vector of the matrix """
        return vector(*(
            self[r][c]
            for r in range(self.shape[0])
        ))

    def pos(self, r, c):
        """ Returns the value at row r and column c """
        return self[r][c]

    @staticmethod
    def zero(rows, cols):
        """ Returns a rows x cols matrix of all zeros """
        return matrix(*(
            vector(*(0 for _ in range(cols)))
            for _ in range(rows)
        ))

    @staticmethod
    def id(n):
        """ Returns an n by n identity matrix """
        if n > 0:
            m = matrix.zero(n, n)
            for i in range(n):
                m[i][i] = 1
            return m
        raise ValueError('n must be at least 1')

    @staticmethod
    def from_numpy(arr):
        """ Convert a 2d np.ndarray to matrix """
        if arr.ndim == 2:
            m = matrix.zero(*arr.shape)
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    m[r][c] = arr[r][c]
            return m
        raise LinAlgError('incompatible shape for matrix')

    def copy(self):
        """ Return a copy of the matrix (row vectors are shallow copied) """
        m = matrix()
        m.shape = self.shape.copy()
        for row in self:
            m.append(row.copy())
        return m

    def __round__(self, n=None):
        """ Round all elements of this matrix to `places` """
        m = self.copy()
        for i, row in enumerate(self):
            m[i] = round(row, n)
        return m

    def __str__(self):
        return '[{}]'.format(', '.join(map(str, self)))

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(repr, self)))

    def latex(self, ctx):
        body = r' \\ '.join(
            r' & '.join(
                x.latex(ctx) if hasattr(x, 'latex') else str(x)
                for x in row
            )
            for row in self
        )
        return r'\begin{bmatrix} ' + body + r' \end{bmatrix}'


_latex_substitutions = {
    # Lower case greek letters
    'α': r'\alpha',
    'β': r'\beta',
    'γ': r'\gamma',
    'δ': r'\delta',
    'ε': r'\varepsilon',
    'ζ': r'\zeta',
    'η': r'\eta',
    'θ': r'\theta',
    'ι': r'\iota',
    'κ': r'\kappa',
    'λ': r'\lambda',
    'μ': r'\mu',
    'ν': r'\nu',
    'ξ': r'\xi',
    'π': r'\pi',
    'ρ': r'\rho',
    'σ': r'\sigma',
    'τ': r'\tau',
    'υ': r'\upsilon',
    'ϕ': r'\phi',
    'φ': r'\varphi',
    'χ': r'\chi',
    'ψ': r'\psi',
    'ω': r'\omega',

    # Upper case greek letters
    'Α': r'A',
    'Β': r'B',
    'Γ': r'\Gamma',
    'Δ': r'\Delta',
    'Ε': r'E',
    'Ζ': r'Z',
    'Η': r'H',
    'Θ': r'\Theta',
    'Ι': r'I',
    'Κ': r'K',
    'Λ': r'\Lambda',
    'Μ': r'M',
    'Ν': r'N',
    'Ξ': r'\Xi',
    'Ο': r'O',
    'Π': r'\Pi',
    'Ρ': r'P',
    'Σ': r'\Sigma',
    'Τ': r'T',
    'Υ': r'\Upsilon',
    'Φ': r'\Phi',
    'Χ': r'X',
    'Ψ': r'\Psi',
    'Ω': r'\Omega',

    # Math symbols
    '∞': r'\infty',
    '∅': r'\emptyset',
    '∂': r'\partial',
    '∇': r'\nabla',
    'ℤ': r'\mathbb{Z}',
    'ℝ': r'\mathbb{R}',
    'ℂ': r'\mathbb{C}',
    'ℕ': r'\mathbb{N}',

    # Misc.
    '_': r'\_',
    '(': r'\left(',
    ')': r'\right)',
    '&': r'\land',
    '|': r'\vee'
}

def replace_latex_symbols(s):
    """
    Replaces all instances of math symbols in a string with their corresponding LaTeX command. Ex.
    ``replace_latex_symbols("test 2π hello") returns "test 2\pi hello". See ``_latex_substitutions`` above. """
    return ''.join(_latex_substitutions.get(c, c) for c in s)
