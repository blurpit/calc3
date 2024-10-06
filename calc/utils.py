import math
import operator
import re
import textwrap
import time
from contextlib import redirect_stdout
from io import StringIO
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
from .parser import parse, Node, ListNode, FunctionCall, Identifier, Declaration, BinaryOperator, UnaryOperator, \
    Function, Variable, Number

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
    if isinstance(expression, str):
        # String expression
        expression = re.sub(r'\s+', '', expression)
        root = parse(ctx, expression)
        answer = root.evaluate(ctx)
    elif isinstance(expression, Node):
        # Expression already parsed into a syntax tree
        answer = expression.evaluate(ctx)
    else:
        # Anything else
        answer = expression

    if isinstance(answer, DeclaredFunction) and answer.is_constant:
        # Evaluate and cache constant values asap
        answer()
    elif isinstance(answer, spread):
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
    else:
        root = expression

    if isinstance(root, DeclaredFunction):
        root = root.func
        msg = 'Declaration ' + str(root)
    elif isinstance(root, Function):
        # Function reference
        definition = ctx.get(root.name)
        if isinstance(definition, DeclaredFunction):
            # If the root is a reference to a DeclaredFunction, print the full tree
            # for the function.
            root = definition.func
            msg = 'Declaration ' + str(definition)
        else:
            # Some other function
            msg = 'Expression ' + str(root)
    elif isinstance(root, Node):
        if isinstance(root, ListNode) and len(root.children) == 1:
            # Remove singleton ListNodes
            root = root.children[0]
        msg = 'Expression ' + str(root)
    else:
        raise TypeError("Can't create a tree from type '{}'".format(type(root).__name__))

    t = treelib.Tree()
    parent = root.parent # temporarily remove parent
    root.parent = None
    root.add_to_tree(t, 0)
    root.parent = parent
    print(msg)
    t.show()

def console(ctx:Context, *, show_time=False, show_tree=False, echo=False):
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
                    # Parse expression
                    exp = re.sub(r'\s+', '', exp)
                    root = parse(ctx, exp)

                    # Print echo or tree
                    if echo:
                        cprint(str(root), Fore.CYAN)
                    if show_tree:
                        tree(ctx, root)

                    # Evaluate expression
                    t = time.perf_counter()
                    result = evaluate(ctx, root)
                    t = time.perf_counter() - t

                    # Add result to context or show graph figure
                    if isinstance(result, DeclaredFunction):
                        ctx.add(result)
                        cprint(f'Added {result.signature} to context.', Fore.YELLOW)
                    elif mpl and plt and isinstance(result, plt.Figure):
                        result.show()
                        continue

                    # Print result
                    cprint(to_str(result), Style.BRIGHT)

                    # Print elapsed time
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
    return _graph(func, xlow, xhigh, ylow, yhigh, n)

def latex(ctx:Context, expression:Union[Definition, str]):
    """ Convert an expression into a LaTeX expression """
    if isinstance(expression, str):
        expression = re.sub(r'\s+', '', expression)
        root = parse(ctx, expression)
    else:
        root = expression

    if isinstance(root, DeclaredFunction):
        root = Declaration(root, root.func)
    elif isinstance(root, FunctionDefinition):
        root = Function(root)
    elif isinstance(root, VariableDefinition):
        root = Variable(root)

    return _latex(ctx, root)


# --- Function implementations --- #

def _concat(a, b):
    """ Merges two lists. Does not preserve `a`! """
    if type(a) != list: a = [a]
    if type(b) == list: a.extend(b)
    else: a.append(b)
    return a

def _concat_all(*args):
    """ Merges a collection of lists """
    result = []
    for item in args:
        if type(item) == list: result.extend(item)
        else: result.append(item)
    return result

def _echo(ctx, x):
    return x

def _help(ctx, obj):
    # Help for ListNode
    if isinstance(obj, ListNode):
        if len(obj.children) == 0:
            return 'help:\nEmpty list'
        else:
            return 'help: {}\nList of {} elements'.format(str(obj), len(obj.children))

    # This generally only happens when the input is a zero-arg
    # function. Extract the function reference itself.
    if isinstance(obj, FunctionCall):
        obj = obj.children[0]

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

def _tree(ctx, root):
    """ Tree function for use in a function definition. Use calc.tree() in regular code. """
    with redirect_stdout(StringIO()) as output:
        tree(ctx, root)

    output.seek(0)
    return output.read().strip()

@replace_none_with_default
def _graph(f, xlow=-10, xhigh=10, ylow=None, yhigh=None, n=1000):
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

def _latex(ctx, root, do_eval=None):
    if do_eval and evaluate(ctx, do_eval):
        result = evaluate(ctx, root)
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

def _scope(f):
    if not isinstance(f, DeclaredFunction):
        raise TypeError('`f` must be a custom defined function')

    s = 'Outer scope of {} {{\n'.format(f.signature)
    if f.saved_scope is not None:
        for definition in f.saved_scope.values():
            s += '\t{}\n'.format(definition)
    s += '}'
    return s

def _del(ctx, obj):
    if not isinstance(obj, Identifier):
        raise TypeError('`obj` must be an identifier')

    ctx.remove(obj.name, DefinitionType.IDENTIFIER)
    return 1

def _and(ctx, a, b):
    return evaluate(ctx, a) and evaluate(ctx, b)

def _or(ctx, a, b):
    return evaluate(ctx, a) or evaluate(ctx, b)

def _greater(a, b):
    return int(a > b)

def _less(a, b):
    return int(a < b)

def _coalesce(ctx, a, b):
    a = evaluate(ctx, a)
    return evaluate(ctx, b) if a is None else a

def _not(x):
    return int(not x)

def _type(obj):
    return type(obj).__name__

def _nroot(x, n):
    return x ** (1/n)

def _hypot(x, y):
    return math.sqrt(x*x + y*y)

def _sec(x):
    return 1 / math.cos(x)

def _csc(x):
    return 1 / math.sin(x)

def _cot(x):
    return 1 / math.tan(x)

def _binomial(p, x, n):
    return math.comb(n, x) * pow(p, x) * pow(1-p, n-x)

def _fibonacci(n):
    return int((_golden**n - (-_golden)**-n) / _sqrt5)

def _randrange(a, b):
    return random() * (b - a) + a

def _average(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return sum(arr) / len(arr)

def _median(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return np.median(arr)

def _sum(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return sum(arr)

def _len(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__len__'):
        arr = arr[0]
    return len(arr)

def _filter(f, *arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return list(filter(f, arr))

def _map(f, *arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return list(map(f, arr))

@replace_none_with_default
def _range(a, b=None, step=1):
    if b is None:
        return list(range(0, a, step))
    else:
        return list(range(a, b, step))

def _max(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return max(arr)

def _min(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        arr = arr[0]
    return min(arr)

def _if(ctx, condition, if_t, if_f):
    if evaluate(ctx, condition):
        return evaluate(ctx, if_t)
    else:
        return evaluate(ctx, if_f)

def _set(*arr):
    if len(arr) == 1 and hasattr(arr[0], '__iter__'):
        return list(set(arr[0]))
    return list(set(arr))

def _shape(M):
    if isinstance(M, matrix):
        return M.shape
    elif isinstance(M, list):
        return [len(M), 1]
    else:
        return [1, 1]

def _print_matrix(ctx, m, align=0.5):
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

def _integrate(f, a, b):
    return sp.integrate.quad(f, a, b)[0]

@replace_none_with_default
def _differentiate(f, x, n=1):
    return sp.misc.derivative(f, x, dx=1e-4, n=n)

def _det(m):
    return np.linalg.det(m)

def _rank(m):
    return np.linalg.matrix_rank(m)

def _nullsp(m):
    return matrix.from_numpy(sp.linalg.null_space(m))

def _rref(m):
    raise NotImplementedError('coming soon to a calculator near you')

def _lu(m):
    l, u = sp.linalg.lu(m, permute_l=True)
    l = matrix.from_numpy(l)
    u = matrix.from_numpy(u)
    return [l, u]

def _svd(m):
    u, s, vh = np.linalg.svd(m)
    u = matrix.from_numpy(u)
    vh = matrix.from_numpy(vh)
    sigma = matrix.zero(*m.shape)
    for i, x in enumerate(s):
        sigma[i][i] = x
    return [u, sigma, vh]

def _cartesian_to_polar(x, y):
    return vector(_hypot(x, y), math.atan2(y, x))

def _polar_to_cartesian(r, theta):
    return vector(r*math.cos(theta), r*math.sin(theta))

def _cartesian_to_cylindrical(x, y, z):
    return vector(_hypot(x, y), math.atan2(y, x), z)

def _cartesian_to_spherical(x, y, z):
    r = math.sqrt(x*x + y*y + z*z)
    return vector(r, math.atan2(y, x), math.acos(z/r))

def _cylindrical_to_cartesian(rho, phi, z):
    return vector(rho*math.cos(phi), rho*math.sin(phi), z)

def _cylindrical_to_spherical(rho, phi, z):
    return vector(_hypot(rho, z), phi, math.atan2(rho, z))

def _spherical_to_cartesian(r, theta, phi):
    return vector(r*math.sin(phi)*math.cos(theta), r*math.sin(theta)*math.sin(phi), r*math.cos(phi))

def _spherical_to_cylindrical(r, theta, phi):
    return vector(r*math.sin(phi), theta, r*math.cos(phi))


# --- LaTeX --- #

def _tex_div(ctx, node, left, right, *_):
    left = left.latex(ctx)
    right = right.latex(ctx)
    return rf'\frac{{{left}}}{{{right}}}'

def _tex_mul(ctx, node, left, right, parens_left, parens_right, is_implicit):
    left = left.latex(ctx)
    right = right.latex(ctx)

    if parens_left:
        left = rf'\left( {left} \right)'
    if parens_right:
        right = rf'\left( {right} \right)'

    if is_implicit:
        return f'{left} {right}'
    else:
        return rf'{left} \cdot {right}'

def _tex_pow(ctx, node, left, right, parens_left, *_):
    left = left.latex(ctx)
    right = right.latex(ctx)
    if parens_left:
        left = rf'\left( {left} \right)'
    return rf'{left}^{{{right}}}'

def _tex_abs(ctx, node, x):
    x = x.latex(ctx)
    return rf'\left| {x} \right|'

def _tex_floor(ctx, node, x):
    x = x.latex(ctx)
    return rf'\lfloor{{{x}}}\rfloor'

def _tex_ceil(ctx, node, x):
    x = x.latex(ctx)
    return rf'\lceil{{{x}}}\rceil'

def _tex_if(ctx, node, condition, if_true, if_false):
    if isinstance(condition, UnaryOperator) and condition.symbol == '!':
        # Condition is negated, remove double negative.
        # Ex. x = 0 instead of !x != 0
        condition = condition.children[0]
        comparator = r'='
    else:
        comparator = r'\neq'

    return rf'\begin{{cases}} ' \
           rf'{if_true.latex(ctx)} & \text{{if }} {condition.latex(ctx)} {comparator} 0 \\' \
           rf'{if_false.latex(ctx)} & \text{{otherwise}} ' \
           rf'\end{{cases}}'

def _tex_root(ctx, node, x, n=None):
    x = x.latex(ctx)
    if n is None:
        return rf'\sqrt{{{x}}}'
    n = n.latex(ctx)
    return rf'\sqrt[{n}]{{{x}}}'

def _tex_log(ctx, node, x, b=None):
    if b is None: b = 10
    else: b = b.latex(ctx)
    return rf'log_{{{b}}}\left( {x} \right)'

def _tex_fact(ctx, node, n):
    if node.is_left_parenthesized(n):
        return rf'\left({{{n.latex(ctx)}}}\right)!'
    return rf'{{{n.latex(ctx)}}}!'

def _tex_choose(ctx, node, n, k):
    return rf'{{{n.latex(ctx)}}}\choose{{{k.latex(ctx)}}}'

def _tex_integral(ctx, node, f, a, b):
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
            body = FunctionCall(f, True)
            body.add_child(Function(definition))
            body.add_child(Variable(ctx.get(differential)))
            body = body.latex(ctx)

        a = a.latex(ctx)
        b = b.latex(ctx)
        differential = replace_latex_symbols(differential)
        return rf'\int_{{{a}}}^{{{b}}} {{{body}}} \, d{differential}'

def _tex_deriv(ctx, node, f, x, n=None):
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
            body = rf'\left( {body} \right)'

        x = x.latex(ctx)
        if n is None:
            frac = rf'\frac{{d}}{{d{differential}}} \Bigr|_{{{differential} = {x}}}'
        else:
            n = n.latex(ctx)
            frac = rf'\frac{{d^{{{n}}}}}{{d{differential}^{{{n}}}}} \Bigr|_{{{differential} = {x}}}'

        return rf'{frac} {{{body}}}'

def _tex_vec(ctx, node, *args):
    return vector(*args).latex(ctx)

def _tex_mat(ctx, node, *args):
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

def _tex_dot(ctx, node, v, w):
    return rf'{v.latex(ctx)} \cdot {w.latex(ctx)}'

def _tex_mag(ctx, node, v):
    return rf'\left\| {v.latex(ctx)} \right\|'

def _tex_mag2(ctx, node, v):
    return rf'{{\left\| {v.latex(ctx)} \right\|}}^{{2}}'

def _tex_transp(ctx, node, m):
    return rf'{{{m.latex(ctx)}}}^{{T}}'


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
        BinaryOperatorDefinition(',', _concat,          0, Associativity.L_TO_R,                 help_text="Concatenation operator"),
        BinaryOperatorDefinition('|', _or,              1, Associativity.L_TO_R,                 help_text="Logical OR operator", manual_eval=True),
        BinaryOperatorDefinition('&', _and,             2, Associativity.L_TO_R,                 help_text="Logical AND operator", manual_eval=True),
        BinaryOperatorDefinition('>', _greater,         3, Associativity.L_TO_R,                 help_text="Greater than comparison"),
        BinaryOperatorDefinition('<', _less,            3, Associativity.L_TO_R,                 help_text="Less than comparison"),
        BinaryOperatorDefinition('+', operator.add,     4, Associativity.L_TO_R,                 help_text="Addition operator"),
        BinaryOperatorDefinition('-', operator.sub,     4, Associativity.L_TO_R,                 help_text="Subtraction operator"),
        BinaryOperatorDefinition('?', _coalesce,        4, Associativity.L_TO_R,                 help_text="Returns `a` if it is not None, otherwise `b`", manual_eval=True),
        BinaryOperatorDefinition('*', operator.mul,     6, Associativity.L_TO_R, latex=_tex_mul, help_text="Multiplication operator"),
        BinaryOperatorDefinition('/', operator.truediv, 6, Associativity.L_TO_R, latex=_tex_div, help_text="Division operator"),
        BinaryOperatorDefinition('%', operator.mod,     6, Associativity.L_TO_R,                 help_text="Remainder operator"),
        BinaryOperatorDefinition('^', operator.pow,     7, Associativity.R_TO_L, latex=_tex_pow, help_text="Exponentiation operator"),

        # Unary operators
        UnaryOperatorDefinition('-', operator.neg, help_text="Unary negation operator"),
        UnaryOperatorDefinition('!', _not,         help_text="Logical NOT operator"),
        UnaryOperatorDefinition('*', spread,       help_text="Unary spread operator. Use it to pass a collection of values as separate arguments."),

        # Basic Functions
        FunctionDefinition('abs',   'x', abs,             latex=_tex_abs,   help_text="Absolute value of `x`"),
        FunctionDefinition('rad',   'θ', math.radians,                      help_text="Converts `θ` in degrees to radians"),
        FunctionDefinition('deg',   'θ', math.degrees,                      help_text="Converts `θ` in radians to degrees"),
        FunctionDefinition('round', 'x', round,                             help_text="Rounds `x` to the nearest integer"),
        FunctionDefinition('floor', 'x', math.floor,      latex=_tex_floor, help_text="Rounds `x` down to the next integer"),
        FunctionDefinition('ceil',  'x', math.ceil,       latex=_tex_ceil,  help_text="Rounds `x` up to the next integer"),
        FunctionDefinition('ans',   '',  lambda: ctx.ans,                   help_text="Answer to the previously evaluated expression"),

        # Informational Functions
        FunctionDefinition('echo',  ['expr'],          _echo,  help_text="Echoes the inputted expression", manual_eval=True, precedence=-99),
        FunctionDefinition('type',  ['obj'],           _type,  help_text="Returns the type of `obj`"),
        FunctionDefinition('help',  ['obj'],           _help,  help_text="Provides a description for the given identifier", manual_eval=True, precedence=-99),
        FunctionDefinition('tree',  ['expr'],          _tree,  help_text="Displays the syntax tree structure of an expression", manual_eval=True, precedence=-99),
        FunctionDefinition('graph', ['f()', 'xlow?', 'xhigh?', 'ylow?', 'yhigh?', 'n?'], _graph, help_text="Graphs a function `f(x)`. `x/y low/high` define the x and y axis scale, and `n` is how many points to evaluate."),
        FunctionDefinition('latex', ['expr', 'eval?'], _latex, help_text="Converts an expression into LaTeX code. Pass 1 to `eval` to evaluate before converting to LaTeX.", manual_eval=True, precedence=0),
        FunctionDefinition('del',   ['obj'],           _del,   help_text="Delete an identifier", manual_eval=True),
        FunctionDefinition('scope', ['f()'],           _scope, help_text="Displays a list of identifiers saved into the scope of a declared function `f`"),

        # Logic & Data Structure Functions
        FunctionDefinition('sum',    ['*x'],                        _sum,                   help_text="Sum of `x`"),
        FunctionDefinition('len',    ['*x'],                        _len,                   help_text="Length of `x`"),
        FunctionDefinition('filter', ['f()', '*x'],                 _filter,                help_text="Filters `x` for elements where `f(x)` is nonzero"),
        FunctionDefinition('map',    ['f()', '*x'],                 _map,                   help_text="Applies a function `f(x)` to each element of `x`"),
        FunctionDefinition('range',  ['a', 'b?', 'step?'],          _range,                 help_text="Returns a list of integers from `a` (inclusive) to `b` (exclusive), or from 0 to `a` if `b` is omitted"),
        FunctionDefinition('max',    ['*x'],                        _max,                   help_text="Returns the largest element of `x`"),
        FunctionDefinition('min',    ['*x'],                        _min,                   help_text="Returns the smallest element of `x`"),
        FunctionDefinition('if',     ['condition', 'if_t', 'if_f'], _if,     latex=_tex_if, help_text="Returns `if_t` if `condition` is nonzero, and `if_f` otherwise", manual_eval=True),
        FunctionDefinition('set',    ['*x'],                        _set,                   help_text="Returns `x` with duplicates removed (order is not preserved)"),

        # Roots & Complex Functions
        FunctionDefinition('sqrt',  'x',  math.sqrt, latex=_tex_root, help_text="Square root of `x`"),
        FunctionDefinition('root',  'xn', _nroot,    latex=_tex_root, help_text="`n`th root of `x`"),
        FunctionDefinition('hypot', 'xy', _hypot,                     help_text="Returns sqrt(x^2 + y^2)"),

        # Trigonometric Functions
        FunctionDefinition('sin',   'θ',  math.sin,   help_text="Sine of `θ` (radians)"),
        FunctionDefinition('cos',   'θ',  math.cos,   help_text="Cosine of `θ` (radians)"),
        FunctionDefinition('tan',   'θ',  math.tan,   help_text="Tangent of `θ` (radians)"),
        FunctionDefinition('sec',   'θ',  _sec,       help_text="Secant of `θ` in radians"),
        FunctionDefinition('csc',   'θ',  _csc,       help_text="Cosecant of `θ` in radians"),
        FunctionDefinition('cot',   'θ',  _cot,       help_text="Cotangent of `θ` in radians"),
        FunctionDefinition('asin',  'x',  math.asin,  help_text="Inverse sine of `x` in radians"),
        FunctionDefinition('acos',  'x',  math.acos,  help_text="Inverse cosine of `x` in radians"),
        FunctionDefinition('atan',  'x',  math.atan,  help_text="Inverse tangent of `x` in radians"),
        FunctionDefinition('atan2', 'xy', math.atan2, help_text="Inverse tangent of `y/x` in radians where the signs of `y` and `x` are considered"),

        # Hyperbolic Functions
        FunctionDefinition('sinh', 'x', math.sinh, help_text="Hyperbolic sine of `x`"),
        FunctionDefinition('cosh', 'x', math.cosh, help_text="Hyperbolic cosine of `x`"),
        FunctionDefinition('tanh', 'x', math.tanh, help_text="Hyperbolic tangent of `x`"),

        # Exponential & Logarithmic Functions
        FunctionDefinition('exp',   'x',  math.exp,                   help_text="Equivalent to `e^x`"),
        FunctionDefinition('ln',    'x',  math.log,                   help_text="Natural logarithm of `x`"),
        FunctionDefinition('log10', 'x',  math.log10, latex=_tex_log, help_text="Base 10 logarithm of `x`"),
        FunctionDefinition('log',   'xb', math.log,   latex=_tex_log, help_text="Base `b` logarithm of `x`"),

        # Combinatorial & Statistics Functions
        FunctionDefinition('fact',   'n',    math.factorial, 7, latex=_tex_fact,   help_text="Factorial of `n`"),
        FunctionDefinition('perm',   'nk',   math.perm,                            help_text="Number of ways to choose `k` items from `n` items without repetition and with order"),
        FunctionDefinition('choose', 'nk',   math.comb,         latex=_tex_choose, help_text="Number of ways to choose `k` items from `n` items without repetition and without order"),
        FunctionDefinition('binom',  'pxn',  _binomial,                            help_text="Probability of an event with probability `p` happening exactly `x` times in `n` trials"),
        FunctionDefinition('fib',    'n',    _fibonacci,                           help_text="`n`th fibonacci number"),
        FunctionDefinition('rand',   '',     random,                               help_text="Random number between 0 and 1"),
        FunctionDefinition('randr',  'ab',   _randrange,                           help_text="Random number between `a` and `b`"),
        FunctionDefinition('avg',    ['*x'], _average,                             help_text="Average of `x`"),
        FunctionDefinition('median', ['*x'], _median,                              help_text="Median of `x`"),

        # Calculus
        FunctionDefinition('int',    ['f()', 'a', 'b'],  _integrate,     latex=_tex_integral, help_text="Definite integral of `f(x)dx` from `a` to `b`"),
        FunctionDefinition('deriv',  ['f()', 'x', 'n?'], _differentiate, latex=_tex_deriv,    help_text="`n`th derivative of `f(x)dx` evaluated at `x`"),

        # Vectors & Matrices
        FunctionDefinition('v',        ['*x'],    vector,        latex=_tex_vec,    help_text="Creates a vector"),
        FunctionDefinition('dot',      'vw',      vector.dot,    latex=_tex_dot,    help_text="Vector dot product"),
        FunctionDefinition('mag',      'v',       vector.mag,    latex=_tex_mag,    help_text="Vector magnitude"),
        FunctionDefinition('mag2',     'v',       vector.mag2,   latex=_tex_mag2,   help_text="Vector magnitude squared"),
        FunctionDefinition('norm',     'v',       vector.norm,                      help_text="Normalizes `v`"),
        FunctionDefinition('zero',     'd',       vector.zero,                      help_text="`d` dimensional zero vector"),
        FunctionDefinition('mat',      ['*rows'], matrix,        latex=_tex_mat,    help_text="Creates a matrix from a list of row vectors"),
        FunctionDefinition('I',        'n',       matrix.id,                        help_text="`n` by `n` identity matrix"),
        FunctionDefinition('shape',    'M',       _shape,                           help_text="Shape of a vector or matrix `M`"),
        FunctionDefinition('mrow',     'Mr',      matrix.row,                       help_text="`r`th row vector of `M`"),
        FunctionDefinition('mcol',     'Mc',      matrix.col,                       help_text="`c`th column vector of `M`"),
        FunctionDefinition('mpos',     'Mrc',     matrix.pos,                       help_text="Value at row `r` and column `c` of `M`"),
        FunctionDefinition('transp',   'M',       matrix.transp, latex=_tex_transp, help_text="Transpose of matrix `M`"),
        FunctionDefinition('printmat', 'M',       _print_matrix,                    help_text="Pretty print a matrix `M`", manual_eval=True),
        FunctionDefinition('vi',       'vi',      vector.i,                         help_text="Value at index `i` of `v`"),

        # Linear Algebra
        FunctionDefinition('det',    'M', _det,    help_text="Determinant of `M`"),
        FunctionDefinition('rank',   'M', _rank,   help_text="Rank of `M`"),
        FunctionDefinition('nullsp', 'M', _nullsp, help_text="Returns a matrix whose columns form a basis for the null space of `M`"),
        FunctionDefinition('rref',   'M', _rref,   help_text="Converts matrix `M` into row-reduced echelon form"),
        FunctionDefinition('lu',     'M', _lu,     help_text="Performs an LU decomposition on matrix `M`, returns (L, U)"),
        FunctionDefinition('svd',    'M', _svd,    help_text="Performs an SVD decomposition on matrix `M`, returns (U, Σ, V^T)"),

        # Coordinate System Conversion Functions
        FunctionDefinition('polar',  'xy',  _cartesian_to_polar,       help_text="Converts 2D cartesian coordinates to 2D polar coordinates"),
        FunctionDefinition('cart',   'rθ',  _polar_to_cartesian,       help_text="Converts 2D polar coordinates to 2D cartesian coordinates"),
        FunctionDefinition('crtcyl', 'xyz', _cartesian_to_cylindrical, help_text="Converts cartesian coordinates to cylindrical coordinates"),
        FunctionDefinition('crtsph', 'xyz', _cartesian_to_spherical,   help_text="Converts cartesian coordinates to spherical coordinates"),
        FunctionDefinition('cylcrt', 'ρϕz', _cylindrical_to_cartesian, help_text="Converts cylindrical coordinates to cartesian coordinates"),
        FunctionDefinition('cylsph', 'ρϕz', _cylindrical_to_spherical, help_text="Converts cylindrical coordinates to spherical coordinates"),
        FunctionDefinition('sphcrt', 'rθϕ', _spherical_to_cartesian,   help_text="Converts spherical coordinates to cartesian coordinates"),
        FunctionDefinition('sphcyl', 'rθϕ', _spherical_to_cylindrical, help_text="Converts spherical coordinates to cylindrical coordinates"),
    )
    return ctx
