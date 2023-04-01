import math
import operator
import re
import textwrap
import time
from contextlib import contextmanager
from random import random
from typing import Union

import numpy as np
import scipy

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    mpl = None
    plt = None

from .context import Context
from .definitions import Associativity, DefinitionType, Definition, FunctionDefinition, VariableDefinition, \
    BinaryOperatorDefinition, UnaryOperatorDefinition, DeclaredFunction, vector, matrix, replace_latex_symbols
from .parser import parse, ListNode, Identifier, Declaration, BinaryOperator, UnaryOperator, Function, Variable

__all__ = ['evaluate', 'tree', 'console', 'graph', 'latex', 'create_default_context']

_golden = 1.618033988749895 # golden ratio (1+√5)/2
_sqrt5 = math.sqrt(5)
class undefined:
    def __str__(self): return 'undefined'
    def __repr__(self): return 'undefined'
_undefined = undefined()

# Setup pyplot style
if mpl and plt:
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rc('font', family='serif')
    plt.rc('text', color='white')
    plt.rc('mathtext', fontset='dejavuserif')
    plt.rc('axes', facecolor='none', edgecolor='none',
           labelsize=28, titlesize=32, labelcolor='white',
           axisbelow=True, grid=True)
    plt.rc('grid', color='#202225', linestyle='solid', lw=3)
    plt.rc('xtick', direction='out', labelsize=18, color='#dcddde')
    plt.rc('ytick', direction='out', labelsize=18, color='#dcddde')
    plt.rc('lines', linewidth=5)
    plt.rc('figure', facecolor='#37393f', figsize=[12, 10], dpi=72)


# --- Calc base --- #

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

    with ctx.with_scope():
        while True:
            exp = input(Fore.YELLOW + '>>> ' + Fore.RESET)
            if exp == 'exit':
                break
            elif exp == 'ctx':
                print(ctx)
            else:
                try:
                    t = time.time()
                    result = evaluate(ctx, exp)
                    t = time.time() - t

                    if isinstance(result, DeclaredFunction):
                        ctx.add(result)
                        cprint(f'Added {result.signature} to context.', Fore.YELLOW)
                    elif type(result) == list:
                        result = ', '.join(map(str, result))
                    elif mpl and plt and isinstance(result, plt.Figure):
                        result.show()
                        continue

                    cprint(str(result), Style.BRIGHT)

                    if show_time:
                        cprint('{:.5f}ms'.format(t*1000), Style.DIM)
                except Exception as e:
                    errprint(e)
            print()

def graph(ctx:Context, func:Union[Definition, str], xlow=-10, xhigh=10, ylow=None, yhigh=None, n=1000):
    """
    Graph a function

    `func` should be a Definition object that represents a 1-dimensional function,
    (takes 1 real input and outputs 1 real output). It can also be a string that
    evaluates to a 1-dimensional function.

    `xlow`, `xhigh`, `ylow`, and `yhigh` determine the range shown on each axis.
    `n` is how many points to compute.
    """
    if isinstance(func, str):
        func = evaluate(ctx, func)
    return graph_(func, xlow, xhigh, ylow, yhigh, n)

def latex(ctx:Context, expression:Union[Definition, str]):
    """ Convert an expression into a LaTeX expression """
    if isinstance(expression, str):
        expression = re.sub(r'\s+', '', expression)
        expression = parse(ctx, expression)

    if isinstance(expression, DeclaredFunction):
        expression = Declaration(expression, expression.func)
    elif isinstance(expression, FunctionDefinition):
        expression = Function(expression)
    elif isinstance(expression, VariableDefinition):
        expression = Variable(expression)

    return latex_(ctx, expression)


# --- Function implementations --- #

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

def tree_(ctx, root):
    """ Tree function for use in a function definition. Use calc.tree() in regular code. """
    import treelib

    with _capture_stdout() as output:
        t = treelib.Tree()

        if isinstance(root, Function) and len(root.children) == 0:
            definition = ctx.get(root.name)
            if isinstance(definition, DeclaredFunction):
                root = definition.func
                msg = 'Declaration ' + str(definition)
            else:
                msg = 'Expression ' + str(root)
        else:
            msg = 'Expression ' + str(root)

        parent = root.parent  # temporarily remove parent
        root.parent = None
        root.add_to_tree(t, 0)
        root.parent = parent

        print(msg)
        t.show()

    output.seek(0)
    return output.read().strip()

def graph_(f, xlow=_undefined, xhigh=_undefined, ylow=_undefined, yhigh=_undefined, n=_undefined):
    """ Graph function for use in a function definition. Use calc.graph() in regular code. """
    if not isinstance(f, Definition):
        raise TypeError("'{}' is not a function".format(f))

    if len(f.args) != 1:
        raise TypeError("{} is not 1-dimensional. Function must take 1 input "
                        "and return 1 output".format(f.signature))

    if xlow is _undefined:
        xlow = -10
    if xhigh is _undefined:
        xhigh = 10
    if ylow is _undefined:
        ylow = None
    if yhigh is _undefined:
        yhigh = None
    if n is _undefined:
        n = 1000

    x = np.linspace(xlow, xhigh, n)
    y = np.empty(len(x))

    # Test the first input for being a float or int
    y[0] = f(x[0])
    if not isinstance(y[0], (float, int)):
        raise TypeError("{} is not 1-dimensional. Function must take 1 input "
                        "and return 1 output".format(f.signature))

    # Fill out the rest of the outputs
    for i in range(1, len(x)):
        result = f(x[i])
        y[i] = float(result)

    fig, ax = plt.subplots(1, 1)

    ax.axhline(0, color='#202225', lw=6)
    ax.axvline(0, color='#202225', lw=6)

    ax.set_xlim(xlow, xhigh)
    if ylow is not None and yhigh is not None:
        ax.set_ylim(ylow, yhigh)
    ax.set_xlabel(str(f.args[0]))
    ax.set_ylabel(str(f.signature))
    # if tex_title:
    #     ax.set_title('${}$'.format(func.latex()))
    # else:
    ax.set_title(textwrap.fill(str(f), 48))

    ax.plot(x, y, color='#ed4245')
    return fig

def latex_(ctx, root):
    parent = root.parent  # temporarily remove parent
    root.parent = None
    result = root.latex(ctx)
    root.parent = parent
    return result

def and_(ctx, a, b):
    return a.evaluate(ctx) and b.evaluate(ctx)

def or_(ctx, a, b):
    return a.evaluate(ctx) or b.evaluate(ctx)

def not_(x):
    return int(x == 0)

def type_(obj):
    return type(obj).__name__

def root(x, n):
    return x ** (1/n)

def hypot(x, y):
    return math.sqrt(x*x + y*y)

def sec(x):
    return 1 / math.cos(x)

def csc(x):
    return 1 / math.sin(x)

def cot(x):
    return 1 / math.tan(x)

def binomial(p, x, n):
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


# --- LaTeX --- #

def tex_div(node, ctx, left, right, *_):
    left = left.latex(ctx)
    right = right.latex(ctx)
    return r'\frac{{{}}}{{{}}}'.format(left, right)

def tex_mul(node, ctx, left, right, parens_left, parens_right, is_implicit):
    left = left.latex(ctx)
    right = right.latex(ctx)

    if parens_left:
        left = r'\left( ' + left + r' \right)'
    if parens_right:
        right = r'\left( ' + right + r' \right)'

    if is_implicit:
        return left + ' ' + right
    else:
        return r'{} \cdot {}'.format(left, right)

def tex_pow(node, ctx, left, right, parens_left, *_):
    left = left.latex(ctx)
    right = right.latex(ctx)
    if parens_left:
        left = r'\left( ' + left + r' \right)'
    return r'{}^{{{}}}'.format(left, right)

def tex_abs(node, ctx, x):
    x = x.latex(ctx)
    return r'\left| ' + x + r' \right|'

def tex_floor(node, ctx, x):
    x = x.latex(ctx)
    return r'\lfloor{' + x + r'}\rfloor'

def tex_ceil(node, ctx, x):
    x = x.latex(ctx)
    return r'\lceil{' + x + r'}\rceil'

def tex_if(node, ctx, condition, if_true, if_false):
    return r'\begin{{cases}} ' \
           r'{} & \text{{if }} {} \neq 0 \\' \
           r'{} & \text{{otherwise}} ' \
           r'\end{{cases}}'.format(
        if_true.latex(ctx),
        condition.latex(ctx),
        if_false.latex(ctx)
    )

def tex_root(node, ctx, x, n=None):
    x = x.latex(ctx)
    if n is None:
        return r'\sqrt{' + x + r'}'
    n = n.latex(ctx)
    return r'\sqrt[{}]{{{}}}'.format(n, x)

def tex_log(node, ctx, x, b=None):
    if b is None: b = 10
    else: b = b.latex(ctx)
    return r'log_{{{}}}\left( {} \right)'.format(b, x)

def tex_fact(node, ctx, n):
    if node.is_left_parenthesized(n):
        return r'\left({{{}}}\right)!'.format(n.latex(ctx))
    return r'{{{}}}!'.format(n.latex(ctx))

def tex_choose(node, ctx, n, k):
    return '{{{}}}\choose{{{}}}'.format(n.latex(ctx), k.latex(ctx))

def tex_integral(node, ctx, f, a, b):
    with ctx.with_scope():
        if isinstance(f, Declaration):
            definition = f.definition
        elif isinstance(f, Function):
            definition = ctx.get(f.name)
        else:
            raise TypeError('Integrand input must be a function')

        if len(definition.args) != 1:
            raise TypeError('Integrand must take 1 argument')

        differential = definition.args[0]
        definition.add_args_to_context(ctx, None)

        if isinstance(f, Declaration):
            body = f.root.latex(ctx)
        else:
            # build a function call
            body = Function(definition)
            body.add_child(Variable(ctx.get(differential)))
            body = body.latex(ctx)

        return r'\int_{{{}}}^{{{}}} {{{}}} \, d{}'.format(
            a.latex(ctx), b.latex(ctx),
            body, replace_latex_symbols(differential)
        )

def tex_deriv(node, ctx, f, x, n=_undefined):
    with ctx.with_scope():
        if isinstance(f, Declaration):
            definition = f.definition
        elif isinstance(f, Function):
            definition = ctx.get(f.name)
        else:
            raise TypeError('Derivative input must be a function')

        if len(definition.args) != 1:
            raise TypeError('Derivative function must take 1 argument')

        differential = definition.args[0]
        definition.add_args_to_context(ctx, None)

        if isinstance(f, Declaration):
            body = f.root.latex(ctx)
            parens_right = node.is_right_parenthesized(f.root)
        else:
            # build a function call
            body = Function(definition)
            body.add_child(Variable(ctx.get(differential)))
            body = body.latex(ctx)
            parens_right = node.is_right_parenthesized(f)

        differential = replace_latex_symbols(differential)
        if parens_right:
            body = r'\left( ' + body + r' \right)'

        if n is _undefined:
            frac = r'\frac{{d}}{{d{}}} \Bigr|_{{{} = {}}}'.format(
                differential, differential, x.latex(ctx)
            )
        else:
            n = n.latex(ctx)
            frac = r'\frac{{d^{{{}}}}}{{d{}^{{{}}}}} \Bigr|_{{{} = {}}}'.format(
                n, differential, n, differential, x.latex(ctx)
            )

        return r'{} {{{}}}'.format(frac, body)

def tex_vec(node, ctx, *args):
    return vector(*args).latex(ctx)

def tex_mat(node, ctx, *args):
    def get_column(arg):
        """ Turns a passed argument (Node object) into a column vector. """
        if isinstance(arg, ListNode):
            # List node "(1,2,3)"
            return vector(*arg.children)
        elif isinstance(arg, Function) and ctx.get(arg.name).func == vector:
            # Vector function call "v(1,2,3)"
            return vector(*arg.children)
        elif isinstance(arg, Variable):
            # Arg is a variable. If the variable's value is a vector, use it
            # as the column.
            val = ctx.get(arg.name).func
            if isinstance(val, vector):
                return val
        return arg

    columns = (get_column(arg) for arg in args)
    return matrix(*columns).latex(ctx)

def tex_dot(node, ctx, v, w):
    return r'{} \cdot {}'.format(v.latex(ctx), w.latex(ctx))

def tex_mag(node, ctx, v):
    return r'\left\| {} \right\|'.format(v.latex(ctx))

def tex_mag2(node, ctx, v):
    return r'{{\left\| {} \right\|}}^{{2}}'.format(v.latex(ctx))

def tex_transp(node, ctx, m):
    return r'{{{}}}^{{T}}'.format(m.latex(ctx))


# --- Context --- #

def create_default_context():
    ctx = Context()
    ctx.add(
        # Constants
        VariableDefinition('π',    math.pi,       help_text="Ratio of a circle's circumference to its diameter"),
        VariableDefinition('pi',   math.pi,  'π', help_text="Ratio of a circle's circumference to its diameter"),
        VariableDefinition('e',    math.e,        help_text="Euler's number"),
        VariableDefinition('ϕ',    _golden,       help_text="The golden ratio"),
        VariableDefinition('phi',  _golden,  'ϕ', help_text="The golden ratio"),
        VariableDefinition('∞',    math.inf,      help_text="Infinity"),
        VariableDefinition('inf',  math.inf, '∞', help_text="Infinity"),
        VariableDefinition('j',    1j,            help_text="Imaginary unit, sqrt(-1)"),
        VariableDefinition('_',    _undefined,    help_text="Undefined value (Can be used to leave certain function arguments as their defaults)"),

        # Binary Operators
        BinaryOperatorDefinition(',', concat,           0, Associativity.L_TO_R,                help_text="Concatenation operator"),
        BinaryOperatorDefinition('+', operator.add,     2, Associativity.L_TO_R,                help_text="Addition operator"),
        BinaryOperatorDefinition('-', operator.sub,     2, Associativity.L_TO_R,                help_text="Subtraction operator"),
        BinaryOperatorDefinition('*', operator.mul,     4, Associativity.L_TO_R, latex=tex_mul, help_text="Multiplication operator"),
        BinaryOperatorDefinition('/', operator.truediv, 4, Associativity.L_TO_R, latex=tex_div, help_text="Division operator"),
        BinaryOperatorDefinition('%', operator.mod,     4, Associativity.L_TO_R,                help_text="Remainder operator"),
        BinaryOperatorDefinition('^', operator.pow,     5, Associativity.R_TO_L, latex=tex_pow, help_text="Exponentiation operator"),
        BinaryOperatorDefinition('&', and_,             4, Associativity.L_TO_R,                help_text="Logical AND operator", manual_eval=True),
        BinaryOperatorDefinition('|', or_,              2, Associativity.L_TO_R,                help_text="Logical OR operator", manual_eval=True),

        # Unary operators
        UnaryOperatorDefinition('-', operator.neg, help_text="Unary negation operator"),
        UnaryOperatorDefinition('!', not_,         help_text="Logical NOT operator"),

        # Basic Functions
        FunctionDefinition('abs',   'x', abs,             latex=tex_abs,   help_text="Absolute value of `x`"),
        FunctionDefinition('rad',   'θ', math.radians,                     help_text="Converts `θ` in degrees to radians"),
        FunctionDefinition('deg',   'θ', math.degrees,                     help_text="Converts `θ` in radians to degrees"),
        FunctionDefinition('round', 'x', round,                            help_text="Rounds `x` to the nearest integer"),
        FunctionDefinition('floor', 'x', math.floor,      latex=tex_floor, help_text="Rounds `x` down to the next integer"),
        FunctionDefinition('ceil',  'x', math.ceil,       latex=tex_ceil,  help_text="Rounds `x` up to the next integer"),
        FunctionDefinition('ans',   '',  lambda: ctx.ans,                  help_text="Answer to the previously evaluated expression"),

        # Informational Functions
        FunctionDefinition('type',  ['obj'],          type_,  help_text="Returns the type of `obj`"),
        FunctionDefinition('help',  ['obj'],          help_,  help_text="Provides a description for the given identifier", manual_eval=True),
        FunctionDefinition('tree',  ['expr'],         tree_,  help_text="Displays the syntax tree structure of an expression", manual_eval=True),
        FunctionDefinition('graph', ['f()', '*args'], graph_, help_text="Graphs a function `f(x)`. `args` includes xlow, xhigh, ylow, yhigh, and n"),
        FunctionDefinition('latex', ['expr'],         latex_, help_text="Converts an expression into LaTeX code", manual_eval=True),

        # Logic & Data Structure Functions
        FunctionDefinition('sum',    ['*x'], sum_,                                     help_text="Sum of `x`"),
        FunctionDefinition('len',    ['*x'], len_,                                     help_text="Length of `x`"),
        FunctionDefinition('filter', ['f()', '*x'], filter_,                           help_text="Filters `x` for elements where `f(x)` is nonzero"),
        FunctionDefinition('range',  ['start', 'stop'], range_,                        help_text="Returns a list of integers from `start` (inclusive) to `stop` (exclusive)"),
        FunctionDefinition('max',    ['*x'], max,                                      help_text="Returns the largest element of `x`"),
        FunctionDefinition('min',    ['*x'], min,                                      help_text="Returns the smallest element of `x`"),
        FunctionDefinition('if',     ['condition', 'if_t', 'if_f'], if_, latex=tex_if, help_text="Returns `if_t` if `condition` is nonzero, and `if_f` otherwise", manual_eval=True),
        FunctionDefinition('set',    ['*x'], set_,                                     help_text="Removes duplicates from a list"),

        # Roots & Complex Functions
        FunctionDefinition('sqrt',  'x',  math.sqrt, latex=tex_root, help_text="Square root of `x`"),
        FunctionDefinition('root',  'xn', root,      latex=tex_root, help_text="`n`th root of `x`"),
        FunctionDefinition('hypot', 'xy', hypot,                     help_text="Returns sqrt(x^2 + y^2)"),

        # Trigonometric Functions
        FunctionDefinition('sin',   'θ',  math.sin,   help_text="Sine of `θ` (radians)"),
        FunctionDefinition('cos',   'θ',  math.cos,   help_text="Cosine of `θ` (radians)"),
        FunctionDefinition('tan',   'θ',  math.tan,   help_text="Tangent of `θ` (radians)"),
        FunctionDefinition('sec',   'θ',  sec,        help_text="Secant of `θ` in radians"),
        FunctionDefinition('csc',   'θ',  csc,        help_text="Cosecant of `θ` in radians"),
        FunctionDefinition('cot',   'θ',  cot,        help_text="Cotangent of `θ` in radians"),
        FunctionDefinition('asin',  'x',  math.asin,  help_text="Inverse sine of `x` in radians"),
        FunctionDefinition('acos',  'x',  math.acos,  help_text="Inverse cosine of `x` in radians"),
        FunctionDefinition('atan',  'x',  math.atan,  help_text="Inverse tangent of `x` in radians"),
        FunctionDefinition('atan2', 'xy', math.atan2, help_text="Inverse tangent of `y/x` in radians where the signs of `y` and `x` are considered"),

        # Hyperbolic Functions
        FunctionDefinition('sinh', 'x', math.sinh, help_text="Hyperbolic sine of `x`"),
        FunctionDefinition('cosh', 'x', math.cosh, help_text="Hyperbolic cosine of `x`"),
        FunctionDefinition('tanh', 'x', math.tanh, help_text="Hyperbolic tangent of `x`"),

        # Exponential & Logarithmic Functions
        FunctionDefinition('exp',   'x',  math.exp,                  help_text="Equivalent to `e^x`"),
        FunctionDefinition('ln',    'x',  math.log,                  help_text="Natural logarithm of `x`"),
        FunctionDefinition('log10', 'x',  math.log10, latex=tex_log, help_text="Base 10 logarithm of `x`"),
        FunctionDefinition('log',   'xb', math.log,   latex=tex_log, help_text="Base `b` logarithm of `x`"),

        # Combinatorial & Random Functions
        FunctionDefinition('fact',   'n',   math.factorial, 7, latex=tex_fact,   help_text="Factorial of `n`"),
        FunctionDefinition('perm',   'nk',  math.perm,                           help_text="Number of ways to choose `k` items from `n` items without repetition and with order"),
        FunctionDefinition('choose', 'nk',  math.comb,         latex=tex_choose, help_text="Number of ways to choose `k` items from `n` items without repetition and without order"),
        FunctionDefinition('binom',  'pxn', binomial,                            help_text="Probability of an event with probability `p` happening exactly `x` times in `n` trials"),
        FunctionDefinition('fib',    'n',   fibonacci,                           help_text="`n`th fibonacci number"),
        FunctionDefinition('rand',   '',    random,                              help_text="Random number between 0 and 1"),
        FunctionDefinition('randr',  'ab',  randrange,                           help_text="Random number between `a` and `b`"),

        # Calculus
        FunctionDefinition('int',    ['f()', 'a', 'b'], integrate,     latex=tex_integral, help_text="Definite integral of `f(x)dx` from `a` to `b`"),
        FunctionDefinition('deriv',  ['f()', 'x'],      differentiate, latex=tex_deriv,    help_text="First derivative of `f(x)dx` evaluated at `x`"),
        FunctionDefinition('nderiv', ['f()', 'x', 'n'], differentiate, latex=tex_deriv,    help_text="`n`th derivative of `f(x)dx` evaluated at `x`"),

        # Vectors & Matrices
        FunctionDefinition('v',      ['*x'],    vector,        latex=tex_vec,    help_text="Creates a vector"),
        FunctionDefinition('dot',    'vw',      vector.dot,    latex=tex_dot,    help_text="Vector dot product"),
        FunctionDefinition('mag',    'v',       vector.mag,    latex=tex_mag,    help_text="Vector magnitude"),
        FunctionDefinition('mag2',   'v',       vector.mag2,   latex=tex_mag2,   help_text="Vector magnitude squared"),
        FunctionDefinition('norm',   'v',       vector.norm,                     help_text="Normalizes `v`"),
        FunctionDefinition('zero',   'd',       vector.zero,                     help_text="`d` dimensional zero vector"),
        FunctionDefinition('mat',    ['*cols'], matrix,        latex=tex_mat,    help_text="Creates a matrix from a list of column vectors"),
        FunctionDefinition('I',      'n',       matrix.id,                       help_text="`n` by `n` identity matrix"),
        FunctionDefinition('shape',  'M',       shape,                           help_text="Shape of a vector or matrix `M`"),
        FunctionDefinition('mrow',   'Mr',      matrix.row,                      help_text="`r`th row vector of `M`"),
        FunctionDefinition('mcol',   'Mc',      matrix.col,                      help_text="`c`th column vector of `M`"),
        FunctionDefinition('mpos',   'Mrc',     matrix.pos,                      help_text="Value at row `r` and column `c` of `M`"),
        FunctionDefinition('transp', 'M',       matrix.transp, latex=tex_transp, help_text="Transpose of matrix `M`"),
        FunctionDefinition('vi',     'vi',      vector.i,                        help_text="Value at index `i` of `v`"),

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
        FunctionDefinition('polar',  'xy',  cartesian_to_polar,       help_text="Converts 2D cartesian coordinates to 2D polar coordinates"),
        FunctionDefinition('cart',   'rθ',  polar_to_cartesian,       help_text="Converts 2D polar coordinates to 2D cartesian coordinates"),
        FunctionDefinition('crtcyl', 'xyz', cartesian_to_cylindrical, help_text="Converts cartesian coordinates to cylindrical coordinates"),
        FunctionDefinition('crtsph', 'xyz', cartesian_to_spherical,   help_text="Converts cartesian coordinates to spherical coordinates"),
        FunctionDefinition('cylcrt', 'ρϕz', cylindrical_to_cartesian, help_text="Converts cylindrical coordinates to cartesian coordinates"),
        FunctionDefinition('cylsph', 'ρϕz', cylindrical_to_spherical, help_text="Converts cylindrical coordinates to spherical coordinates"),
        FunctionDefinition('sphcrt', 'rθϕ', spherical_to_cartesian,   help_text="Converts spherical coordinates to cartesian coordinates"),
        FunctionDefinition('sphcyl', 'rθϕ', spherical_to_cylindrical, help_text="Converts spherical coordinates to cylindrical coordinates"),

        override_global=True
    )
    return ctx

