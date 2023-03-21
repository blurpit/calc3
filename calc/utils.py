import math
import operator
import re
import time
from contextlib import contextmanager
from random import random

import scipy

from .context import Context
from .definitions import Associativity, DefinitionType, FunctionDefinition, VariableDefinition, \
    BinaryOperatorDefinition, UnaryOperatorDefinition, DeclaredFunction, vector, matrix
from .parser import parse, Identifier, Declaration, BinaryOperator, UnaryOperator, Function

__all__ = ['evaluate', 'tree', 'console', 'create_global_context']

_golden = 1.618033988749895 # golden ratio (1+√5)/2
_sqrt5 = math.sqrt(5)

def evaluate(ctx:Context, expression:str):
    """ Evaluate an expression """
    expression = re.sub(r'\s+', '', expression)
    root = parse(ctx, expression)

    answer = root.evaluate(ctx)
    if isinstance(answer, DeclaredFunction) and answer.is_constant:
        # Evaluate and cache constant values asap
        answer()

    ctx.ans = answer
    return answer

def tree(ctx:Context, expression:str):
    """ Parse an expression and print the syntax tree structure. """
    import treelib
    expression = re.sub(r'\s+', '', expression)
    root = parse(ctx, expression)

    t = treelib.Tree()
    root.add_to_tree(t, 0)
    print('Expression', str(root))
    t.show()

def console(ctx:Context, show_time=False):
    """ Start an interactive console """
    # noinspection PyUnresolvedReferences
    from colorama import Fore, Style, just_fix_windows_console
    just_fix_windows_console()

    def cprint(s, col):
        print(col + str(s) + Style.RESET_ALL)
    def errprint(exc):
        cprint('{}: {}'.format(type(exc).__name__, str(exc)), Fore.RED)

    debug = False
    with ctx.with_scope():
        while True:
            exp = input(Fore.YELLOW + '>>> ' + Fore.RESET)
            if exp == 'exit':
                break
            elif exp == 'reset_ctx':
                ctx.pop_scope()
                ctx.push_scope()
            elif exp == '!':
                debug = True
            else:
                try:
                    t = time.time()
                    result = evaluate(ctx, exp)
                    # result = latex(ctx, exp)
                    t = time.time() - t

                    if isinstance(result, DeclaredFunction):
                        ctx.add(result)

                    if type(result) == list:
                        result = ', '.join(map(str, result))
                    cprint(str(result), Style.BRIGHT)

                    if show_time:
                        cprint('{:.5f}ms'.format(t*1000), Style.DIM)
                except Exception as e:
                    errprint(e)
            print()

@contextmanager
def _capture_stdout():
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    string = StringIO()
    try:
        sys.stdout = string
        yield string
    finally:
        sys.stdout = old_stdout

def concat(a, b):
    """ Concatenates two objects into a list """
    if type(a) != list: a = [a]
    if type(b) == list: a.extend(b)
    else: a.append(b)
    return a

def concat_all(*args):
    result = []
    for item in args:
        if type(item) == list: result.extend(item)
        else: result.append(item)
    return result

def help_(ctx, obj):
    # Help for identifiers
    if isinstance(obj, Identifier):
        definition = ctx.get(obj.name)

        if definition.is_constant:
            signature = '{} = {}'.format(definition.signature, definition.func)
        elif isinstance(definition, DeclaredFunction):
            signature = str(definition)
        else:
            signature = definition.signature

        if definition.help_text is not None:
            help_text = definition.help_text
        elif not definition.is_constant and definition.func.__doc__ is not None:
            help_text = definition.func.__doc__
        elif isinstance(definition, DeclaredFunction):
            help_text = "User defined function or constant"
        else:
            help_text = "No description provided"

        return 'help: {}\n{}'.format(signature, help_text)

    # Help for declared functions
    if isinstance(obj, Declaration):
        return 'help: {}\nUser defined function or constant'.format(str(obj))

    # Help for binary operators
    if isinstance(obj, BinaryOperator):
        definition = ctx.get(obj.symbol, DefinitionType.BINARY_OPERATOR)
        return 'help: {}\n{}'.format(
            definition.name,
            definition.help_text or "No description provided"
        )

    # Help for unary operators
    if isinstance(obj, UnaryOperator):
        definition = ctx.get(obj.symbol, DefinitionType.UNARY_OPERATOR)
        return 'help: {}\n{}'.format(
            definition.name,
            definition.help_text or "No description provided"
        )

    return 'help: {}\nNo description provided'.format(obj)

def tree_(ctx, obj):
    # not to be confused with tree()
    import treelib

    with _capture_stdout() as output:
        t = treelib.Tree()

        if isinstance(obj, Function) and len(obj.children) == 0:
            definition = ctx.get(obj.name)
            if isinstance(definition, DeclaredFunction):
                obj = definition.func
                msg = 'Declaration ' + str(definition)
        else:
            msg = 'Expression ' + str(obj)

        parent = obj.parent  # temporarily remove parent
        obj.parent = None
        obj.add_to_tree(t, 0)
        obj.parent = parent

        print(msg)
        t.show()

    output.seek(0)
    return output.read().strip()

def and_(ctx, a, b):
    return a.evaluate(ctx) and b.evaluate(ctx)

def or_(ctx, a, b):
    return a.evaluate(ctx) or b.evaluate(ctx)

def type_(obj):
    return type(obj).__name__

def hypot(x, y):
    return math.sqrt(x*x + y*y)

def binomial(n, x, p):
    return math.comb(n, x) * pow(p, x) * pow(1-p, n-x)

def fibonacci(n):
    return int((_golden**n - (-_golden)**-n) / _sqrt5)

def randrange(a, b):
    return random() * (b - a) + a

def sum_(*arr):
    if len(arr) == 1:
        return sum(arr[0])
    return sum(arr)

def len_(*arr):
    if len(arr) == 1 and isinstance(arr[0], list):
        return len(arr[0])
    return len(arr)

def filter_(f, *arr):
    if len(arr) == 1 and isinstance(arr[0], list):
        arr = arr[0]
    return list(filter(f, arr))

def range_(start, stop):
    return list(range(start, stop))

def if_(ctx, condition, if_t, if_f):
    if condition.evaluate(ctx):
        return if_t.evaluate(ctx)
    else:
        return if_f.evaluate(ctx)

def set_(*arr):
    if len(arr) == 1 and isinstance(arr[0], list):
        return list(set(arr[0]))
    return list(set(arr))

def shape(M):
    if isinstance(M, matrix):
        return M.shape
    elif isinstance(M, list):
        return [len(M), 1]
    else:
        return [1, 1]

def integrate(f, a, b):
    return scipy.integrate.quad(f, a, b)[0]

def differentiate(f, x, n=1):
    return scipy.misc.derivative(f, x, dx=1e-4, n=n)

def cartesian_to_polar(x, y):
    return vector(hypot(x, y), math.atan2(y, x))

def polar_to_cartesian(r, theta):
    return vector(r*math.cos(theta), r*math.sin(theta))

def cartesian_to_cylindrical(x, y, z):
    return vector(hypot(x, y), math.atan2(y, x), z)

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    return vector(r, math.atan2(y, x), math.acos(z/r))

def cylindrical_to_cartesian(rho, phi, z):
    return vector(rho*math.cos(phi), rho*math.sin(phi), z)

def cylindrical_to_spherical(rho, phi, z):
    return vector(hypot(rho, z), phi, math.atan2(rho, z))

def spherical_to_cartesian(r, theta, phi):
    return vector(r*math.sin(phi)*math.cos(theta), r*math.sin(theta)*math.sin(phi), r*math.cos(phi))

def spherical_to_cylindrical(r, theta, phi):
    return vector(r*math.sin(phi), theta, r*math.cos(phi))


def create_global_context():
    ctx = Context()
    ctx.add(
        # Constants
        VariableDefinition('π',   math.pi,       help_text="Ratio of a circle's circumference to its diameter"),
        VariableDefinition('pi',  math.pi, 'π',  help_text="Ratio of a circle's circumference to its diameter"),
        VariableDefinition('e',   math.e,        help_text="Euler's number"),
        VariableDefinition('ϕ',   _golden,       help_text="The golden ratio"),
        VariableDefinition('phi', _golden, 'ϕ',  help_text="The golden ratio"),
        VariableDefinition('∞',   math.inf,      help_text="Infinity"),
        VariableDefinition('inf', math.inf, '∞', help_text="Infinity"),
        VariableDefinition('j',   1j,            help_text="Imaginary unit, sqrt(-1)"),

        # Binary Operators
        BinaryOperatorDefinition(',', concat,           0, Associativity.L_TO_R, help_text="Concatenation operator"),
        BinaryOperatorDefinition('+', operator.add,     2, Associativity.L_TO_R, help_text="Addition operator"),
        BinaryOperatorDefinition('-', operator.sub,     2, Associativity.L_TO_R, help_text="Subtraction operator"),
        BinaryOperatorDefinition('*', operator.mul,     4, Associativity.L_TO_R, help_text="Multiplication operator"),
        BinaryOperatorDefinition('/', operator.truediv, 4, Associativity.L_TO_R, help_text="Division operator"),
        BinaryOperatorDefinition('%', operator.mod,     4, Associativity.L_TO_R, help_text="Remainder operator"),
        BinaryOperatorDefinition('^', operator.pow,     6, Associativity.R_TO_L, help_text="Exponentiation operator"),
        BinaryOperatorDefinition('&', and_,             4, Associativity.L_TO_R, help_text="Logical AND operator", manual_eval=True),
        BinaryOperatorDefinition('|', or_,              2, Associativity.L_TO_R, help_text="Logical OR operator", manual_eval=True),

        # Unary operators
        UnaryOperatorDefinition('-', operator.neg, help_text="Unary negation operator"),

        # Basic Functions
        FunctionDefinition('neg',   'x', operator.neg,    help_text="Negates `x`"),
        FunctionDefinition('abs',   'x', abs,             help_text="Absolute value of `x`"),
        FunctionDefinition('rad',   'θ', math.radians,    help_text="Converts `θ` in degrees to radians"),
        FunctionDefinition('deg',   'θ', math.degrees,    help_text="Converts `θ` in radians to degrees"),
        FunctionDefinition('round', 'x', round,           help_text="Rounds `x` to the nearest integer"),
        FunctionDefinition('floor', 'x', math.floor,      help_text="Rounds `x` down to the next integer"),
        FunctionDefinition('ceil',  'x', math.ceil,       help_text="Rounds `x` up to the next integer"),
        FunctionDefinition('ans',   '',  lambda: ctx.ans, help_text="Answer to the previously evaluated expression"),

        # Informational Functions
        FunctionDefinition('type',  ['obj'],  type_, help_text="Gets the type of `obj`"),
        FunctionDefinition('help',  ['obj'],  help_, help_text="Provide a description for the given object", manual_eval=True),
        FunctionDefinition('tree',  ['expr'], tree_, help_text="Display the syntax tree structure", manual_eval=True),

        # Logic & Data structure functions
        FunctionDefinition('sum',    ['*x'], sum_,                       help_text="Sum of `x`"),
        FunctionDefinition('len',    ['*x'], len_,                       help_text="Length of `x`"),
        FunctionDefinition('filter', ['f()', '*x'], filter_,             help_text="Filter `x` for elements where `f` evaluates to true"),
        FunctionDefinition('range',  ['start', 'stop'], range_,          help_text="List of integers from `start` (inclusive) to `stop` (exclusive)"),
        FunctionDefinition('max',    ['*x'], max,                        help_text="Returns the largest element of `x`"),
        FunctionDefinition('min',    ['*x'], min,                        help_text="Returns the smallest element of `x`"),
        FunctionDefinition('if',     ['condition', 'if_t', 'if_f'], if_, help_text="Returns `if_t` if `condition` is nonzero, and `if_f` otherwise", manual_eval=True),
        FunctionDefinition('set',    ['*x'], set_,                       help_text="Removes duplicates from a list"),

        # Roots & Complex Functions
        FunctionDefinition('sqrt',  'x',  math.sqrt,             help_text="Square root of `x`"),
        FunctionDefinition('root',  'xn', lambda x, n: x**(1/n), help_text="`n`th root of `x`"),
        FunctionDefinition('hypot', 'xy', lambda x, y: hypot,    help_text="Returns sqrt(x^2 + y^2)"),

        # Trigonometric Functions
        FunctionDefinition('sin',   'θ',  math.sin,                help_text="Sine of `θ` (radians)"),
        FunctionDefinition('cos',   'θ',  math.cos,                help_text="Cosine of `θ` (radians)"),
        FunctionDefinition('tan',   'θ',  math.tan,                help_text="Tangent of `θ` (radians)"),
        FunctionDefinition('sec',   'θ',  lambda x: 1/math.cos(x), help_text="Secant of `θ` in radians"),
        FunctionDefinition('csc',   'θ',  lambda x: 1/math.sin(x), help_text="Cosecant of `θ` in radians"),
        FunctionDefinition('cot',   'θ',  lambda x: 1/math.tan(x), help_text="Cotangent of `θ` in radians"),
        FunctionDefinition('asin',  'x',  math.asin,               help_text="Inverse sine of `x` in radians"),
        FunctionDefinition('acos',  'x',  math.acos,               help_text="Inverse cosine of `x` in radians"),
        FunctionDefinition('atan',  'x',  math.atan,               help_text="Inverse tangent of `x` in radians"),
        FunctionDefinition('atan2', 'xy', math.atan2,              help_text="Inverse tangent of `y/x` in radians where the signs of `y` and `x` are considered"),

        # Hyperbolic Functions
        FunctionDefinition('sinh', 'x', math.sinh, help_text="Hyperbolic sine of `x`"),
        FunctionDefinition('cosh', 'x', math.cosh, help_text="Hyperbolic cosine of `x`"),
        FunctionDefinition('tanh', 'x', math.tanh, help_text="Hyperbolic tangent of `x`"),

        # Exponential & Logarithmic Functions
        FunctionDefinition('exp',   'x',  math.exp,   help_text="Equal to `e^x`"),
        FunctionDefinition('ln',    'x',  math.log,   help_text="Natural logarithm of `x`"),
        FunctionDefinition('log10', 'x',  math.log10, help_text="Base 10 logarithm of `x`"),
        FunctionDefinition('log',   'xb', math.log,   help_text="Base `b` logarithm of `x`"),

        # Combinatorial & Random Functions
        FunctionDefinition('fact',   'n',   math.factorial, help_text="Factorial of `n`"),
        FunctionDefinition('perm',   'nk',  math.perm,      help_text="Number of permutations for selecting `k` items from a set of `n` total items"),
        FunctionDefinition('choose', 'nk',  math.comb,      help_text="Number of combinations for selecting `k` items from a set of `n` total items"),
        FunctionDefinition('binom',  'nxp', binomial,       help_text="Probability of an event with probability `p` happening exactly `x` times in `n` trials"),
        FunctionDefinition('fib',    'n',   fibonacci,      help_text="`n`th fibonacci number"),
        FunctionDefinition('rand',   '',    random,         help_text="Random real number between 0 and 1"),
        FunctionDefinition('randr',  'ab',  randrange,      help_text="Random real number between `a` and `b`"),

        # Calculus
        FunctionDefinition('int',    ['f()', 'a', 'b'], integrate,     help_text="Definite integral of `f(x)dx` from `a` to `b`"),
        FunctionDefinition('deriv',  ['f()', 'x'],      differentiate, help_text="First derivative of `f(x)dx` evaluated at `x`"),
        FunctionDefinition('nderiv', ['f()', 'x', 'n'], differentiate, help_text="`n`th derivative of `f(x)dx` evaluated at `x`"),

        # Vectors & Matrices
        FunctionDefinition('v',     ['*x'],    vector,      help_text="Creates a vector"),
        FunctionDefinition('dot',   'vw',      vector.dot,  help_text="Vector dot product"),
        FunctionDefinition('mag',   'v',       vector.mag,  help_text="Vector magnitude"),
        FunctionDefinition('mag2',  'v',       vector.mag2, help_text="Vector magnitude squared"),
        FunctionDefinition('norm',  'v',       vector.norm, help_text="Normalizes `v`"),
        FunctionDefinition('zero',  'd',       vector.zero, help_text="`d` dimensional zero vector"),
        FunctionDefinition('mat',   ['*cols'], matrix,      help_text="Creates a matrix from a set of column vectors"),
        FunctionDefinition('I',     'n',       matrix.id,   help_text="`n` by `n` identity matrix"),
        FunctionDefinition('shape', 'M',       shape,       help_text="Shape of a vector or matrix `M`"),
        FunctionDefinition('mrow',  'Mr',      matrix.row,  help_text="`r`th row vector of `M`"),
        FunctionDefinition('mcol',  'Mc',      matrix.col,  help_text="`c`th column vector of `M`"),
        FunctionDefinition('mpos',  'Mrc',     matrix.pos,  help_text="Value at row `r` and column `c` of `M`"),
        FunctionDefinition('vi',    'vi',      vector.i,    help_text="Value at index `i` of `v`"),

        # Linear Algebra
        # FunctionDefinition('det', 'M', determinant),
        # FunctionDefinition('rank', 'M', rank),
        # FunctionDefinition('inv', 'M', invert),
        # FunctionDefinition('kernel', 'M', kernel),
        # FunctionDefinition('ech', 'M', echelon),
        # FunctionDefinition('isech', 'M', is_echelon),
        # FunctionDefinition('rref', 'M', rref),
        # FunctionDefinition('isrref', 'M', is_rref),
        # FunctionDefinition('lu', 'M', lu),
        # FunctionDefinition('svd', 'M', svd),

        # Coordinate System Conversion Functions
        FunctionDefinition('polar',  'xy',  cartesian_to_polar,       help_text="Convert 2D cartesian coordinates to 2D polar coordinates"),
        FunctionDefinition('cart',   'rθ',  polar_to_cartesian,       help_text="Convert 2D polar coordinates to 2D cartesian coordinates"),
        FunctionDefinition('crtcyl', 'xyz', cartesian_to_cylindrical, help_text="Convert cartesian coordinates to cylindrical coordinates"),
        FunctionDefinition('crtsph', 'xyz', cartesian_to_spherical,   help_text="Convert cartesian coordinates to spherical coordinates"),
        FunctionDefinition('cylcrt', 'ρϕz', cylindrical_to_cartesian, help_text="Convert cylindrical coordinates to cartesian coordinates"),
        FunctionDefinition('cylsph', 'ρϕz', cylindrical_to_spherical, help_text="Convert cylindrical coordinates to spherical coordinates"),
        FunctionDefinition('sphcrt', 'rθϕ', spherical_to_cartesian,   help_text="Convert spherical coordinates to cartesian coordinates"),
        FunctionDefinition('sphcyl', 'rθϕ', spherical_to_cylindrical, help_text="Convert spherical coordinates to cylindrical coordinates"),

        override_global=True
    )
    return ctx

