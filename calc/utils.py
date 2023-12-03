import math
import operator
import re
import textwrap
import time
from contextlib import contextmanager
from random import random
from typing import Union

import numpy as np
import scipy as sp

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    mpl = None
    plt = None

from .context import Context
from .definitions import Associativity, DefinitionType, Definition, FunctionDefinition, VariableDefinition, \
    BinaryOperatorDefinition, UnaryOperatorDefinition, DeclaredFunction, vector, matrix, spread, replace_latex_symbols
from .parser import parse, Node, ListNode, Identifier, Declaration, BinaryOperator, UnaryOperator, Function, Variable, \
    Number

__all__ = [
    'evaluate',
    'tree',
    'console',
    'graph',
    'latex',
    'create_default_context',
    'replace_none_with_default'
]


# --- Helpers --- #

_golden = 1.618033988749895 # golden ratio (1+√5)/2
_sqrt5 = math.sqrt(5)

@contextmanager
def _capture_stdout():
    """ Capture anything sent to stdout inside a with block and save it to a BytesIO """
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    string = StringIO()
    try:
        sys.stdout = string
        yield string
    finally:
        sys.stdout = old_stdout

def replace_none_with_default(f):
    """
    Function wrapper that replaces ``None`` with defaults in a function. Example::

        @calc.replace_none_with_default
        def foo(a, b=1, c=2, d=3):
            print(a, b, c, d)

        ctx.add(FunctionDefinition('foo', 'abcd', foo))
        calc.evaluate(ctx, 'foo(_, _, _, _)')

    Will print "None 1 2 3"
    """
    def wrapper(*args):
        args = list(args)
        # number of required positional args
        offset = f.__code__.co_argcount - len(f.__defaults__)
        for i in range(offset, len(args)):
            if args[i] is None:
                args[i] = f.__defaults__[i - offset]
        return f(*args)
    return wrapper

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

def evaluate(ctx:Context, expression):
    """ Evaluate an expression """
    if isinstance(expression, Node):
        return expression.evaluate(ctx)
    elif not isinstance(expression, str):
        return expression

    expression = re.sub(r'\s+', '', expression)
    root = parse(ctx, expression)

    answer = root.evaluate(ctx)
    if isinstance(answer, DeclaredFunction) and answer.is_constant:
        # Evaluate and cache constant values asap
        answer()
    if isinstance(answer, spread):
        # Get rid of spread operators
        answer = list(answer)

    answer = ctx.round_result(answer)
    ctx.ans = answer
    return answer

def tree(ctx:Context, expression:Union[Definition, str]):
    """ Parse an expression and print the syntax tree structure. """
    import treelib

    if isinstance(expression, str):
        expression = re.sub(r'\s+', '', expression)
        root = parse(ctx, expression)
        if len(root.children) == 1:
            # Remove singleton ListNodes
            root = root.children[0]
    else:
        root = expression

    if isinstance(root, DeclaredFunction):
        root = root.func
        msg = 'Declaration ' + str(root)
    elif isinstance(root, Function) and len(root.children) == 0:
        # Function reference
        definition = ctx.get(root.name)
        if isinstance(definition, DeclaredFunction):
            # DeclaredFunction reference
            root = definition.func
            msg = 'Declaration ' + str(definition)
        else:
            # Some other function
            msg = 'Expression ' + str(root)
    else:
        msg = 'Expression ' + str(root)

    t = treelib.Tree()
    parent = root.parent # temporarily remove parent
    root.parent = None
    root.add_to_tree(t, 0)
    root.parent = parent
    print(msg)
    t.show()

def console(ctx:Context, *, show_time=False, show_tree=False):
    """ Start an interactive console """
    # noinspection PyUnresolvedReferences
    from colorama import Fore, Style, just_fix_windows_console
    just_fix_windows_console()

    def cprint(s, col):
        """ Print a message with a given color """
        print(col + str(s) + Style.RESET_ALL)
    def errprint(exc):
        """ Print an exception """
        cprint('{}: {}'.format(type(exc).__name__, str(exc)), Fore.RED)
    def to_str(x, depth=0):
        if type(x) == list:
            s = ', '.join(to_str(x2, depth+1) for x2 in x)
            if depth > 0:
                s = '(' + s + ')'
            return s
        return str(x)

    with ctx.with_scope():
        while True:
            exp = input(Fore.YELLOW + '>>> ' + Fore.RESET)

            if exp == 'exit':
                break
            elif exp == 'ctx':
                print(ctx)
            else:
                try:
                    if show_tree:
                        tree(ctx, exp)
                    t = time.perf_counter()
                    result = evaluate(ctx, exp)
                    t = time.perf_counter() - t

                    if isinstance(result, DeclaredFunction):
                        ctx.add(result)
                        cprint(f'Added {result.signature} to context.', Fore.YELLOW)
                    elif mpl and plt and isinstance(result, plt.Figure):
                        result.show()
                        continue

                    cprint(to_str(result), Style.BRIGHT)

                    if show_time:
                        cprint('{:.5f}ms'.format(t*1000), Style.DIM)
                except Exception as e:
                    errprint(e)
                except KeyboardInterrupt:
                    pass
            print()

def graph(ctx:Context, func:Union[Definition, str], xlow=-10, xhigh=10, ylow=None, yhigh=None, n=1000):
    """
    Graph a function

    `func` should be a Definition object that represents a 1-dimensional function (takes 1 real input and returns 1
    real output). It can also be a string that evaluates to a 1-dimensional function.

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

def concat(a, b):
    """ Merges two lists. Does not preserve `a`! """
    if type(a) != list: a = [a]
    if type(b) == list: a.extend(b)
    else: a.append(b)
    return a

def concat_all(*args):
    """ Merges a collection of lists """
    result = []
    for item in args:
        if type(item) == list: result.extend(item)
        else: result.append(item)
    return result

def help_(ctx, obj):
    # Help for ListNode
    if isinstance(obj, ListNode):
        if len(obj.children) == 0:
            return 'help:\nEmpty list'
        else:
            return 'help: {}\nList of {} elements'.format(str(obj), len(obj.children))

    # Help for identifiers
    if isinstance(obj, Identifier):
        if isinstance(obj, Declaration):
            # Definition is inside the declaration
            definition = obj.definition
        else:
            # Definition is inside the context
            definition = ctx.get(obj.name)

        if definition.is_constant:
            # Show full definition for constants (name + value)
            signature = '{} = {}'.format(definition.signature, definition.func)
        elif isinstance(definition, DeclaredFunction):
            # Show full definition for declared functions (signature + body)
            signature = str(definition)
        else:
            # Show only the signature for generic FunctionDefinitions
            signature = definition.signature

        if definition.help_text is not None:
            # Use help text given by the definition
            help_text = definition.help_text
        elif not definition.is_constant and definition.func.__doc__ is not None:
            # Use the function's documentation as help text
            help_text = definition.func.__doc__.strip()
        elif isinstance(definition, DeclaredFunction):
            # Generic help text for declared functions
            if definition.is_constant:
                help_text = "User defined constant"
            else:
                help_text = "User defined function"
        else:
            # FunctionDefinition with no help text
            help_text = "No description provided"

        return 'help: {}\n{}'.format(signature, help_text)

    # Help for binary operators
    if isinstance(obj, BinaryOperator):
        definition = ctx.get(obj.symbol, DefinitionType.BINARY_OPERATOR)
        return 'help: {}\n{}'.format(
            definition.signature,
            definition.help_text or "No description provided"
        )

    # Help for unary operators
    if isinstance(obj, UnaryOperator):
        definition = ctx.get(obj.symbol, DefinitionType.UNARY_OPERATOR)
        return 'help: {}\n{}'.format(
            definition.signature,
            definition.help_text or "No description provided"
        )

    # Help for numbers
    if isinstance(obj, Number):
        return 'help: {}\nNumber literal'.format(obj)

    # Everything else
    return 'help: {}\nNo description provided'.format(obj)

def tree_(ctx, root):
    """ Tree function for use in a function definition. Use calc.tree() in regular code. """
    with _capture_stdout() as output:
        tree(ctx, root)

    output.seek(0)
    return output.read().strip()

@replace_none_with_default
def graph_(f, xlow=-10, xhigh=10, ylow=None, yhigh=None, n=1000):
    """ Graph function for use in a function definition. Use calc.graph() in regular code. """
    if not mpl or not plt:
        raise ImportError('matplotlib is required to use graphing')

    if not isinstance(f, Definition):
        raise TypeError("'{}' is not a function".format(f))

    if len(f.args) != 1:
        raise TypeError("{} is not 1-dimensional. Function must take 1 input and return 1 output".format(f.signature))

    # Make x and y-axis arrays
    x = np.linspace(xlow, xhigh, n)
    y = np.empty(len(x))

    # Fill out the rest of the outputs
    for i in range(len(x)):
        try:
            result = f(x[i])
            y[i] = float(result)
        except Exception:
            y[i] = np.nan

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

def latex_(ctx, root, do_eval=None):
    if do_eval and evaluate(ctx, do_eval):
        result = evaluate(ctx, root)
        result = ctx.round_result(result)
        def to_latex(obj):
            if hasattr(obj, 'latex'):
                return obj.latex(ctx)
            elif isinstance(obj, list):
                return r',\, '.join(map(to_latex, obj))
            else:
                return replace_latex_symbols(str(obj))
        return to_latex(result)

    parent = root.parent  # temporarily remove parent
    root.parent = None
    result = root.latex(ctx)
    root.parent = parent
    return result

def scope_(f):
    if not isinstance(f, DeclaredFunction):
        raise TypeError('`f` must be a custom defined function')

    s = 'Outer scope of {} {{\n'.format(f.signature)
    if f.saved_scope is not None:
        for definition in f.saved_scope.values():
            if not isinstance(definition, DeclaredFunction) and definition.is_constant:
                s += '\t{} = {}\n'.format(definition, definition())
            else:
                s += '\t{}\n'.format(definition)
    s += '}'
    return s

def del_(ctx, obj):
    if not isinstance(obj, Identifier):
        raise TypeError('`obj` must be an identifier')

    ctx.remove(obj.name, DefinitionType.IDENTIFIER)
    return 1

def and_(ctx, a, b):
    return evaluate(ctx, a) and evaluate(ctx, b)

def or_(ctx, a, b):
    return evaluate(ctx, a) or evaluate(ctx, b)

def greater(a, b):
    return int(a > b)

def less(a, b):
    return int(a < b)

def coalesce(ctx, a, b):
    a = evaluate(ctx, a)
    return evaluate(ctx, b) if a is None else a

def not_(x):
    return int(not x)

def type_(obj):
    return type(obj).__name__

def nroot(x, n):
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

def average(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return sum(arr) / len(arr)

def median(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return np.median(arr)

def sum_(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return sum(arr)

def len_(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__len__'):
        arr = arr[0]
    return len(arr)

def filter_(f, *arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return list(filter(f, arr))

def map_(f, *arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return list(map(f, arr))

@replace_none_with_default
def range_(a, b=None, step=1):
    if b is None:
        return list(range(0, a, step))
    else:
        return list(range(a, b, step))

def max_(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return max(arr)

def min_(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return min(arr)

def if_(ctx, condition, if_t, if_f):
    if evaluate(ctx, condition):
        return evaluate(ctx, if_t)
    else:
        return evaluate(ctx, if_f)

def set_(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        return list(set(arr[0]))
    return list(set(arr))

def shape(M):
    if isinstance(M, matrix):
        return M.shape
    elif isinstance(M, list):
        return [len(M), 1]
    else:
        return [1, 1]

def print_matrix(ctx, m, align=0.5):
    m = evaluate(ctx, m)
    align = evaluate(ctx, align)

    if not isinstance(m, matrix):
        raise TypeError('`M` must be a matrix')
    if not isinstance(align, (int, float)):
        raise TypeError('`align` must be a number')

    align = max(0.0, min(align, 1.0))
    m = ctx.round_result(m)

    # find lengths of each column
    col_lens = [0] * m.shape[1]
    for row in m:
        for i, x in enumerate(row):
            col_lens[i] = max(col_lens[i], len(str(x)))

    lines = []
    for r, row in enumerate(m):
        line = []
        for c, x in enumerate(row):
            s = str(x)
            space = col_lens[c] - len(s)
            l_sp = math.ceil(space * align)
            r_sp = math.floor(space * (1 - align))
            s = ' ' * l_sp + s + ' ' * r_sp
            line.append(s)

        if len(m) == 1:
            # 1-row matrix
            line.insert(0, '[')
            line.append(']')
        elif r == 0:
            # top row
            line.insert(0, '⎡')
            line.append('⎤')
        elif r < len(m) - 1:
            # middle row
            line.insert(0, '⎢')
            line.append('⎥')
        else:
            # bottom row
            line.insert(0, '⎣')
            line.append('⎦')

        lines.append('  '.join(line))

    return '\n'.join(lines)

def integrate(f, a, b):
    return sp.integrate.quad(f, a, b)[0]

@replace_none_with_default
def differentiate(f, x, n=1):
    return sp.misc.derivative(f, x, dx=1e-4, n=n)

def det(m):
    return np.linalg.det(m)

def rank(m):
    return np.linalg.matrix_rank(m)

def nullsp(m):
    return matrix.from_numpy(sp.linalg.null_space(m))

def rref(m):
    raise NotImplementedError('coming soon to a calculator near you')

def lu(m):
    l, u = sp.linalg.lu(m, permute_l=True)
    l = matrix.from_numpy(l)
    u = matrix.from_numpy(u)
    return [l, u]

def svd(m):
    u, s, vh = np.linalg.svd(m)
    u = matrix.from_numpy(u)
    vh = matrix.from_numpy(vh)
    sigma = matrix.zero(*m.shape)
    for i, x in enumerate(s):
        sigma[i][i] = x
    return [u, sigma, vh]

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

def tex_div(ctx, node, left, right, *_):
    left = left.latex(ctx)
    right = right.latex(ctx)
    return r'\frac{{{}}}{{{}}}'.format(left, right)

def tex_mul(ctx, node, left, right, parens_left, parens_right, is_implicit):
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

def tex_pow(ctx, node, left, right, parens_left, *_):
    left = left.latex(ctx)
    right = right.latex(ctx)
    if parens_left:
        left = r'\left( ' + left + r' \right)'
    return r'{}^{{{}}}'.format(left, right)

def tex_abs(ctx, node, x):
    x = x.latex(ctx)
    return r'\left| ' + x + r' \right|'

def tex_floor(ctx, node, x):
    x = x.latex(ctx)
    return r'\lfloor{' + x + r'}\rfloor'

def tex_ceil(ctx, node, x):
    x = x.latex(ctx)
    return r'\lceil{' + x + r'}\rceil'

def tex_if(ctx, node, condition, if_true, if_false):
    if isinstance(condition, UnaryOperator) and condition.symbol == '!':
        # Condition is negated, remove double negative.
        # Ex. x = 0 instead of !x != 0
        condition = condition.children[0]
        comparator = r'='
    else:
        comparator = r'\neq'

    return r'\begin{{cases}} ' \
           r'{} & \text{{if }} {} {} 0 \\' \
           r'{} & \text{{otherwise}} ' \
           r'\end{{cases}}'.format(
        if_true.latex(ctx),
        condition.latex(ctx),
        comparator,
        if_false.latex(ctx)
    )

def tex_root(ctx, node, x, n=None):
    x = x.latex(ctx)
    if n is None:
        return r'\sqrt{' + x + r'}'
    n = n.latex(ctx)
    return r'\sqrt[{}]{{{}}}'.format(n, x)

def tex_log(ctx, node, x, b=None):
    if b is None: b = 10
    else: b = b.latex(ctx)
    return r'log_{{{}}}\left( {} \right)'.format(b, x)

def tex_fact(ctx, node, n):
    if node.is_left_parenthesized(n):
        return r'\left({{{}}}\right)!'.format(n.latex(ctx))
    return r'{{{}}}!'.format(n.latex(ctx))

def tex_choose(ctx, node, n, k):
    return '{{{}}}\choose{{{}}}'.format(n.latex(ctx), k.latex(ctx))

def tex_integral(ctx, node, f, a, b):
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

def tex_deriv(ctx, node, f, x, n=None):
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

        if n is None:
            frac = r'\frac{{d}}{{d{}}} \Bigr|_{{{} = {}}}'.format(
                differential, differential, x.latex(ctx)
            )
        else:
            n = n.latex(ctx)
            frac = r'\frac{{d^{{{}}}}}{{d{}^{{{}}}}} \Bigr|_{{{} = {}}}'.format(
                n, differential, n, differential, x.latex(ctx)
            )

        return r'{} {{{}}}'.format(frac, body)

def tex_vec(ctx, node, *args):
    return vector(*args).latex(ctx)

def tex_mat(ctx, node, *args):
    def get_column(arg):
        """ Turns a passed argument (Node object) into a column vector. """
        if isinstance(arg, ListNode) or isinstance(arg, Function) and ctx.get(arg.name).func == vector:
            # List node "(1,2,3)" or vector function call "v(1,2,3)"
            return vector(*arg.children)
        elif isinstance(arg, Variable):
            # Arg is a variable. If the variable's value is a vector, use it as the column.
            val = ctx.get(arg.name).func
            if isinstance(val, vector):
                return val
        return arg

    columns = (get_column(arg) for arg in args)
    return matrix(*columns).latex(ctx)

def tex_dot(ctx, node, v, w):
    return r'{} \cdot {}'.format(v.latex(ctx), w.latex(ctx))

def tex_mag(ctx, node, v):
    return r'\left\| {} \right\|'.format(v.latex(ctx))

def tex_mag2(ctx, node, v):
    return r'{{\left\| {} \right\|}}^{{2}}'.format(v.latex(ctx))

def tex_transp(ctx, node, m):
    return r'{{{}}}^{{T}}'.format(m.latex(ctx))


# --- Context --- #

def create_default_context():
    ctx = Context()
    ctx.add_global(
        # Constants
        VariableDefinition('π',    math.pi,       help_text="Ratio of a circle's circumference to its diameter"),
        VariableDefinition('pi',   math.pi,  'π', help_text="Ratio of a circle's circumference to its diameter"),
        VariableDefinition('e',    math.e,        help_text="Euler's number"),
        VariableDefinition('ϕ',    _golden,       help_text="The golden ratio"),
        VariableDefinition('phi',  _golden,  'ϕ', help_text="The golden ratio"),
        VariableDefinition('∞',    math.inf,      help_text="Infinity"),
        VariableDefinition('inf',  math.inf, '∞', help_text="Infinity"),
        VariableDefinition('j',    1j,            help_text="Imaginary unit, sqrt(-1)"),
        VariableDefinition('_',    None,          help_text="None value"),

        # Binary Operators
        BinaryOperatorDefinition(',', concat,           0, Associativity.L_TO_R,                help_text="Concatenation operator"),
        BinaryOperatorDefinition('|', or_,              1, Associativity.L_TO_R,                help_text="Logical OR operator", manual_eval=True),
        BinaryOperatorDefinition('&', and_,             2, Associativity.L_TO_R,                help_text="Logical AND operator", manual_eval=True),
        BinaryOperatorDefinition('>', greater,          3, Associativity.L_TO_R,                help_text="Greater than comparison"),
        BinaryOperatorDefinition('<', less,             3, Associativity.L_TO_R,                help_text="Less than comparison"),
        BinaryOperatorDefinition('+', operator.add,     4, Associativity.L_TO_R,                help_text="Addition operator"),
        BinaryOperatorDefinition('-', operator.sub,     4, Associativity.L_TO_R,                help_text="Subtraction operator"),
        BinaryOperatorDefinition('?', coalesce,         4, Associativity.L_TO_R,                help_text="Returns `a` if it is not None, otherwise `b`", manual_eval=True),
        BinaryOperatorDefinition('*', operator.mul,     6, Associativity.L_TO_R, latex=tex_mul, help_text="Multiplication operator"),
        BinaryOperatorDefinition('/', operator.truediv, 6, Associativity.L_TO_R, latex=tex_div, help_text="Division operator"),
        BinaryOperatorDefinition('%', operator.mod,     6, Associativity.L_TO_R,                help_text="Remainder operator"),
        BinaryOperatorDefinition('^', operator.pow,     7, Associativity.R_TO_L, latex=tex_pow, help_text="Exponentiation operator"),

        # Unary operators
        UnaryOperatorDefinition('-', operator.neg, help_text="Unary negation operator"),
        UnaryOperatorDefinition('!', not_,         help_text="Logical NOT operator"),
        UnaryOperatorDefinition('*', spread,       help_text="Unary spread operator. Use it to pass a collection of values as separate arguments."),

        # Basic Functions
        FunctionDefinition('abs',   'x', abs,             latex=tex_abs,   help_text="Absolute value of `x`"),
        FunctionDefinition('rad',   'θ', math.radians,                     help_text="Converts `θ` in degrees to radians"),
        FunctionDefinition('deg',   'θ', math.degrees,                     help_text="Converts `θ` in radians to degrees"),
        FunctionDefinition('round', 'x', round,                            help_text="Rounds `x` to the nearest integer"),
        FunctionDefinition('floor', 'x', math.floor,      latex=tex_floor, help_text="Rounds `x` down to the next integer"),
        FunctionDefinition('ceil',  'x', math.ceil,       latex=tex_ceil,  help_text="Rounds `x` up to the next integer"),
        FunctionDefinition('ans',   '',  lambda: ctx.ans,                  help_text="Answer to the previously evaluated expression"),

        # Informational Functions
        FunctionDefinition('type',  ['obj'],           type_,  help_text="Returns the type of `obj`"),
        FunctionDefinition('help',  ['obj'],           help_,  help_text="Provides a description for the given identifier", manual_eval=True, precedence=-99),
        FunctionDefinition('tree',  ['expr'],          tree_,  help_text="Displays the syntax tree structure of an expression", manual_eval=True, precedence=-99),
        FunctionDefinition('graph', ['f()', 'xlow?', 'xhigh?', 'ylow?', 'yhigh?', 'n?'], graph_, help_text="Graphs a function `f(x)`. `x/y low/high` define the x and y axis scale, and `n` is how many points to evaluate."),
        FunctionDefinition('latex', ['expr', 'eval?'], latex_, help_text="Converts an expression into LaTeX code. Pass 1 to `eval` to evaluate before converting to LaTeX.", manual_eval=True, precedence=0),
        FunctionDefinition('del',   ['obj'],           del_,   help_text="Delete an identifier", manual_eval=True),
        FunctionDefinition('scope', ['f()'],           scope_, help_text="Displays a list of identifiers saved into the scope of a declared function `f`"),

        # Logic & Data Structure Functions
        FunctionDefinition('sum',    ['*x'],                        sum_,                  help_text="Sum of `x`"),
        FunctionDefinition('len',    ['*x'],                        len_,                  help_text="Length of `x`"),
        FunctionDefinition('filter', ['f()', '*x'],                 filter_,               help_text="Filters `x` for elements where `f(x)` is nonzero"),
        FunctionDefinition('map',    ['f()', '*x'],                 map_,                  help_text="Applies a function `f(x)` to each element of `x`"),
        FunctionDefinition('range',  ['a', 'b?', 'step?'],          range_,                help_text="Returns a list of integers from `a` (inclusive) to `b` (exclusive), or from 0 to `a` if `b` is omitted"),
        FunctionDefinition('max',    ['*x'],                        max_,                  help_text="Returns the largest element of `x`"),
        FunctionDefinition('min',    ['*x'],                        min_,                  help_text="Returns the smallest element of `x`"),
        FunctionDefinition('if',     ['condition', 'if_t', 'if_f'], if_,     latex=tex_if, help_text="Returns `if_t` if `condition` is nonzero, and `if_f` otherwise", manual_eval=True),
        FunctionDefinition('set',    ['*x'],                        set_,                  help_text="Returns `x` with duplicates removed (order is not preserved)"),

        # Roots & Complex Functions
        FunctionDefinition('sqrt',  'x',  math.sqrt, latex=tex_root, help_text="Square root of `x`"),
        FunctionDefinition('root',  'xn', nroot,     latex=tex_root, help_text="`n`th root of `x`"),
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

        # Combinatorial & Statistics Functions
        FunctionDefinition('fact',   'n',    math.factorial, 7, latex=tex_fact,   help_text="Factorial of `n`"),
        FunctionDefinition('perm',   'nk',   math.perm,                           help_text="Number of ways to choose `k` items from `n` items without repetition and with order"),
        FunctionDefinition('choose', 'nk',   math.comb,         latex=tex_choose, help_text="Number of ways to choose `k` items from `n` items without repetition and without order"),
        FunctionDefinition('binom',  'pxn',  binomial,                            help_text="Probability of an event with probability `p` happening exactly `x` times in `n` trials"),
        FunctionDefinition('fib',    'n',    fibonacci,                           help_text="`n`th fibonacci number"),
        FunctionDefinition('rand',   '',     random,                              help_text="Random number between 0 and 1"),
        FunctionDefinition('randr',  'ab',   randrange,                           help_text="Random number between `a` and `b`"),
        FunctionDefinition('avg',    ['*x'], average,                             help_text="Average of `x`"),
        FunctionDefinition('median', ['*x'], median,                              help_text="Median of `x`"),

        # Calculus
        FunctionDefinition('int',    ['f()', 'a', 'b'],  integrate,     latex=tex_integral, help_text="Definite integral of `f(x)dx` from `a` to `b`"),
        FunctionDefinition('deriv',  ['f()', 'x', 'n?'], differentiate, latex=tex_deriv,    help_text="`n`th derivative of `f(x)dx` evaluated at `x`"),

        # Vectors & Matrices
        FunctionDefinition('v',        ['*x'],    vector,        latex=tex_vec,    help_text="Creates a vector"),
        FunctionDefinition('dot',      'vw',      vector.dot,    latex=tex_dot,    help_text="Vector dot product"),
        FunctionDefinition('mag',      'v',       vector.mag,    latex=tex_mag,    help_text="Vector magnitude"),
        FunctionDefinition('mag2',     'v',       vector.mag2,   latex=tex_mag2,   help_text="Vector magnitude squared"),
        FunctionDefinition('norm',     'v',       vector.norm,                     help_text="Normalizes `v`"),
        FunctionDefinition('zero',     'd',       vector.zero,                     help_text="`d` dimensional zero vector"),
        FunctionDefinition('mat',      ['*rows'], matrix,        latex=tex_mat,    help_text="Creates a matrix from a list of row vectors"),
        FunctionDefinition('I',        'n',       matrix.id,                       help_text="`n` by `n` identity matrix"),
        FunctionDefinition('shape',    'M',       shape,                           help_text="Shape of a vector or matrix `M`"),
        FunctionDefinition('mrow',     'Mr',      matrix.row,                      help_text="`r`th row vector of `M`"),
        FunctionDefinition('mcol',     'Mc',      matrix.col,                      help_text="`c`th column vector of `M`"),
        FunctionDefinition('mpos',     'Mrc',     matrix.pos,                      help_text="Value at row `r` and column `c` of `M`"),
        FunctionDefinition('transp',   'M',       matrix.transp, latex=tex_transp, help_text="Transpose of matrix `M`"),
        FunctionDefinition('printmat', 'M',       print_matrix,                    help_text="Pretty print a matrix `M`", manual_eval=True),
        FunctionDefinition('vi',       'vi',      vector.i,                        help_text="Value at index `i` of `v`"),

        # Linear Algebra
        FunctionDefinition('det',    'M', det,    help_text="Determinant of `M`"),
        FunctionDefinition('rank',   'M', rank,   help_text="Rank of `M`"),
        FunctionDefinition('nullsp', 'M', nullsp, help_text="Returns a matrix whose columns form a basis for the null space of `M`"),
        FunctionDefinition('rref',   'M', rref,   help_text="Converts matrix `M` into row-reduced echelon form"),
        FunctionDefinition('lu',     'M', lu,     help_text="Performs an LU decomposition on matrix `M`, returns (L, U)"),
        FunctionDefinition('svd',    'M', svd,    help_text="Performs an SVD decomposition on matrix `M`, returns (U, Σ, V^T)"),

        # Coordinate System Conversion Functions
        FunctionDefinition('polar',  'xy',  cartesian_to_polar,       help_text="Converts 2D cartesian coordinates to 2D polar coordinates"),
        FunctionDefinition('cart',   'rθ',  polar_to_cartesian,       help_text="Converts 2D polar coordinates to 2D cartesian coordinates"),
        FunctionDefinition('crtcyl', 'xyz', cartesian_to_cylindrical, help_text="Converts cartesian coordinates to cylindrical coordinates"),
        FunctionDefinition('crtsph', 'xyz', cartesian_to_spherical,   help_text="Converts cartesian coordinates to spherical coordinates"),
        FunctionDefinition('cylcrt', 'ρϕz', cylindrical_to_cartesian, help_text="Converts cylindrical coordinates to cartesian coordinates"),
        FunctionDefinition('cylsph', 'ρϕz', cylindrical_to_spherical, help_text="Converts cylindrical coordinates to spherical coordinates"),
        FunctionDefinition('sphcrt', 'rθϕ', spherical_to_cartesian,   help_text="Converts spherical coordinates to cartesian coordinates"),
        FunctionDefinition('sphcyl', 'rθϕ', spherical_to_cylindrical, help_text="Converts spherical coordinates to cylindrical coordinates"),
    )
    return ctx
