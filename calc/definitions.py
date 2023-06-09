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
    precedence = 3
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
        self.manual_eval = kwargs.get('manual_eval', False)
        self.help_text = kwargs.get('help_text', None)

    def check_inputs(self, n_inputs):
        if self.manual_eval:
            # manual eval funcs have an extra ctx input
            n_inputs -= 1

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
        else:
            return self.func(*inputs)

    def add_args_to_context(self, ctx, inputs):
        """ Define args in a context as unknown definitions. """
        if inputs is None:
            # Unknown signature; match each arg to None
            for i in range(len(self.args)):
                ctx.add(_argument_definition_wrapper(self.args[i], self.f_args[i], None))
        elif self.star_arg:
            # Match each arg until the star arg, then pass the star arg as a list of the
            # remaining inputs
            for i in range(len(self.args)-1):
                ctx.add(_argument_definition_wrapper(self.args[i], self.f_args[i], inputs[i]))
            last = len(self.args) - 1
            ctx.add(_argument_definition_wrapper(self.args[last], self.f_args[last], inputs[last:]))
        else:
            # Match each arg to each input
            for i in range(len(self.args)):
                ctx.add(_argument_definition_wrapper(self.args[i], self.f_args[i], inputs[i]))

    @property
    def signature(self):
        if self.is_constant:
            return self.display_name or self.name

        def argname(i):
            name = self.args[i]
            if self.star_arg and i == len(self.args)-1:
                name = '*' + name
            if self.f_args[i]:
                name = name + '()'
            return name

        return '{}({})'.format(
            self.display_name or self.name,
            ', '.join(map(argname, range(len(self.args))))
        )

    @property
    def is_constant(self):
        """ Constant functions take 0 arguments and their `func` is a value instead of
            a callable. """
        return len(self.args) == 0 and not callable(self.func)

    def __str__(self):
        return self.signature

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.name, self.args)

def _argument_definition_wrapper(name, is_func, value):
    """
    Returns a Definition for functions whose full signature is unknown. This is added
    to the context at parse-time so the parser knows a given identifier exists before
    knowing its exact definition. Kind of like a forward declaration in C. Kind of.

    :param name: Identifier name
    :param is_func: Whether this identifier takes more than 1 or more inputs
    :param value: Value of the identifier (or None if unknown)
    """
    if is_func:
        return Definition(name, ('...',), (False,), value, star_arg=True)
    else:
        return Definition(name, (), (), value)


class FunctionDefinition(Definition):
    def __init__(self, name, args, func, precedence=None, display_name=None,
                 latex=None, help_text=None, manual_eval=False):
        """
        Define a function

        `args` should be a list of strings which represent the name of each argument.
        If this function takes another function as an argument, add () to the name of
        that argument, for example ``args=['f()', 'x']`` takes a function `f` and a
        variable `x` as inputs.

        `func` should be a callable if the function takes 1 or more arguments, otherwise
        it may be either a callable or a direct value.

        `precedence` is used when the function is called implicitly to determine what
        goes inside the parentheses. Default is 3 (between + and *). Ex. "sin2*pi/3+8"
        is parsed as "sin(2*pi/3)+8. Higher number = higher precedence.

        Set `display_name` to have the expression show a different name from what was
        inputted when parsed and converted back to a string. Ex. the funcion declaration
        "f(x)=2pix" will be shown as "f(x) = 2πx" when printed.

        `latex` should be a function that that takes `node`, `ctx`, and `*inputs`. `node` is
        the ``Function`` node in the syntax tree representing the function call, and `*inputs`
        are the inputs to the function call. It should return a LaTeX string. For example, for
        the expression ``int(sin, 0, 3)``, `node` will be a ``Function`` object where ``node.name``
        is ``'int'`` (you can get ``int``'s the definition using ``ctx.get(node.name)``), and
        ``inputs`` will be ``(Number(0), Number(3))``. Use ``inputs[i].latex(ctx)`` to convert an
        input to latex. Passing a `latex` function is useful for special functions that should be
        rendered differently from the standard ``name(a, b, c)`` format in latex, integrals for
        example.

        `help_text` will be shown when evaluating ``help([name])``. If none is provided, it
        will use the docstring associated with `func`. If that is also not provided, it will
        say "No description provided."

        Inputs passed into `func` are usually pre-evaluated. If `manual_eval` is True,
        then they will not be evaluated, and instead `func` will be passed first the
        context, then the ``Node`` objects representing the inputs to the function. Use
        ``.evaluate(ctx)`` to evaluate them manually. This can be useful for some functions
        that need to short-circuit, such as ``if(condition, if_t, if_f)``, where if the
        condition is true, ``false`` should not be evaluated.

        :param name: Name of the function
        :param args: List of argument names
        :param func: The function itself
        :param precedence: Precedence for this function if parentheses are absent (default 2)
        :param display_name: Name to use for str(definition)
        :param latex: Function that converts a function call expression to LaTeX
        :param help_text: Text to return when evaluating help(...)
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
        # If func is None we'll assume it's supposed to be replaced later. If someone doesn't
        # replace it, that's more their problem than mine.
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
    ``f = calc.evaluate(ctx, "f(x) = 3x^2")`` will store a DeclaredFunction object into
    ``f``. Declared functions can be called like a normal function (``f(4)`` will return
    48) or added to a context using ``ctx.add(f)``.
    """

    def __init__(self, name, args, is_const):
        super().__init__(name, args, None)
        self.ctx = None
        self._is_const = is_const and len(args) == 0

    def copy(self):
        """ Make a shallow copy of this definition object """
        return copy(self)

    def bind_context(self, ctx):
        """ Bind this function to a context that will be used when __call__ is invoked """
        self.ctx = ctx

    def __call__(self, *inputs):
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
            # add parentheses if the function returns a list "f(x)=(1,2,3), 4"
            body = '(' + body + ')'
        return '{} = {}'.format(self.signature, body)


class BinaryOperatorDefinition(Definition):
    token_type = DefinitionType.BINARY_OPERATOR

    def __init__(self, symbol, func, precedence, associativity, latex=None, help_text=None, manual_eval=False):
        """
        Define a binary operator

        `func` should be a callable that takes 2 inputs and returns the result of the operator,
        where the first input is the left operand and the 2nd is the left.

        `precedence` and `associativity` are required for binary operators. Precedence defines
        what order operators are evaluated when parentheses aren't used. Higher number means
        higher precedence.

        `latex` should be a function that takes the following inputs: `self`, `ctx`, `left`, `right`,
        `parens_left`, `parens_right`, and `is_implicit`. `self` is this BinaryOperatorDefinition
        object, `left` and `right` are the operands (Node objects). `parens_right` and `parens_left`
        say whether the left/right operands are parenthesized (this happens when they are lower
        precedence than the parent operator, ex. (3+5)/2). `is_implicit` says whether the operator
        is implicit, ex. 2*π vs 2π. Use ``.latex(ctx)`` to convert `left` and `right` to latex.

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


class UnaryOperatorDefinition(Definition):
    precedence = 5
    associativity = Associativity.R_TO_L
    token_type = DefinitionType.UNARY_OPERATOR

    def __init__(self, symbol, func, precedence=None, latex=None, help_text=None):
        """
        Define a unary operator.

        :param symbol: Operator symbol (must be exactly 1 character)
        :param func: Function that implements the operator (should take 1 input)
        :param precedence: Operator precedence
        :param help_text: Text shown on help()
        """
        if len(symbol) != 1:
            raise ValueError("Invalid unary operator symbol '{}'; "
                             "sybmols must be 1 character".format(symbol))
        super().__init__(
            symbol, 'x', (False,), func,
            help_text=help_text
        )
        if precedence is not None:
            self.precedence = precedence
        self.latex_func = latex

    def __str__(self):
        return self.name + 'x'


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

    def round(self, places):
        """ Round all elements of this vector to `places` """
        for i, x in enumerate(self):
            self[i] = round(x, places)
        return self

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
                other * column
                for column in self
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
                v = vector.zero(len(other))
                for c in range(self.shape[1]):
                    for r in range(self.shape[0]):
                        v[r] += other[r] * self[r][c]
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
                    arr = linalg.inv(self)
                    return matrix.from_numpy(arr)
                elif power == 0:
                    return matrix.id(self.shape[0])
                elif power > 0:
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

    def copy(self):
        return matrix(*(row.copy() for row in self))

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
        if arr.ndim != 2:
            raise LinAlgError('incompatible shape for matrix')
        m = matrix.zero(*arr.shape)
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                m[r][c] = arr[r][c]
        return m

    def round(self, places):
        """ Round all elements of this matrix to `places` """
        for row in self:
            row.round(places)
        return self

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
}

def replace_latex_symbols(s):
    result = ''
    for c in s:
        result += _latex_substitutions.get(c, c)
    return result
