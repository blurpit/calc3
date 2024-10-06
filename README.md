# Calculator

An overly complicated calculator I made out of boredom (and for fun). The 
original purpose of this project was to have something neat to put into blurbot,
my discord bot, where people could evaluate math expressions using the bot.
Although this never fully made it into the bot, I feel like I learned a good bit
about tree structures, text parsing, math, etc. by doing this.

This is my third attempt at making an expression parser. The code in the first
two versions was horrible, so I made a third one out of spite, because I 
couldn't let this project defeat me. This project suffers from a serious case
of feature creep.


## Install
```
pip install git+https://github.com/blurpit/calc3.git
```

Required dependencies: numpy, scipy, colorama, treelib. Optional dependencies: 
matplotlib.


# Table of Contents
- [Usage](#usage)
- [Expression evaluation](#expression-evaluation)
  * [Simple expressions](#simple-expressions)
  * [Using functions](#using-functions)
    + [Implicit calls](#implicit-calls)
    + [Zero-argument calls](#zero-argument-calls)
  * [Creating functions](#creating-functions)
  * [Creating variables](#creating-variables)
  * [Function references](#function-references)
    + [Functions as arguments](#functions-as-arguments)
  * [Recursion](#recursion)
  * [Advanced function creation](#advanced-function-creation)
    + [Function arguments](#function-arguments)
    + [Optional arguments](#optional-arguments)
    + [Star arguments](#star-arguments)
    + [Combining argument types](#combining-argument-types)
  * [Lists](#lists)
    + [Spread operator](#spread-operator)
  * [Vectors & matrices](#vectors--matrices)
- [Graphing](#graphing)
- [LaTeX](#latex)
- [Scopes & contexts](#scopes--contexts)
    + [Definitions](#definitions)
    + [Shadowing](#shadowing)
    + [Global scope](#global-scope)
      - [Global scope modification](#global-scope-modification)
      - [Global scope shadowing](#global-scope-shadowing)
    + [Settings](#settings)
    + [The default context](#the-default-context)
      - [Constants](#constants)
      - [Binary operators](#binary-operators)
      - [Unary operators](#unary-operators)
      - [Basic functions](#basic-functions)
      - [Informational functions](#informational-functions)
      - [Logic & data structure functions](#logic--data-structure-functions)
      - [Roots & complex functions](#roots--complex-functions)
      - [Trigonometric functions](#trigonometric-functions)
      - [Hyperbolic functions](#hyperbolic-functions)
      - [Exponential & logarithmic functions](#exponential--logarithmic-functions)
      - [Combinatorial & statistics functions](#combinatorial--statistics-functions)
      - [Calculus](#calculus)
      - [Vectors & matrices](#vectors--matrices-1)
      - [Linear algebra](#linear-algebra)
      - [Coordinate system conversion functions](#coordinate-system-conversion-functions)
- [Python definitions](#python-definitions)
    + [Binary operators](#binary-operators-1)
    + [Unary operators](#unary-operators-1)
    + [Variables](#variables)
    + [Other arguments](#other-arguments)
- [Evaluation details](#evaluation-details)
    + [Precedence & Associativity](#precedence--associativity)
    + [Syntax trees](#syntax-trees)
    + [Saved scopes](#saved-scopes)
      - [Usage example](#usage-example)
      - [Saved scope settings](#saved-scope-settings)
      - [Returning functions](#returning-functions)
- [Saving to JSON](#saving-to-json)


# Usage
```python
import calc
ctx = calc.create_default_context()
ctx.push_scope()
```
The [context](#scopes--contexts) contains definitions for all binary operators, 
functions, variables, and constants. The rest of this document will be using 
the [default context](#the-default-context) for its examples.

Use `calc.evaluate()` to evaluate an expression, or `calc.console()` to get an 
interactive console.

```python
answer = calc.evaluate(ctx, '1+1')
print(answer)
> 2
```

```python
calc.console(ctx)
>>> 1+1
2
```
*This document uses examples in both python and `calc.console`. If you see 
`>>>`, it's a console input. Lines that start with `>` are print statement 
outputs.*


# Expression evaluation
The meat and potatoes of any expression evaluator, really.

## Simple expressions
```python
# Simple addition
>>> 9 + 10
19

# Order of operations and parentheses
>>> 2^(3+1)/12+1.5
2.833333333333333

# Using constants
>>> e^-3 + e^2
7.438843167298513

# Multiple outputs example
>>> 2^1, 2^2, 2^3, 2^4, 2^5, 2^6
2, 4, 8, 16, 32, 64

# Implicit multiplication example
>>> 4(3+2)(6)
120

# Whitespace is ignored when evaluating expressions
>>> (1    +          4  )     * 7
35
```

## Using functions

Functions are used the same as in math or any standard programming language, 
with the function name followed by the arguments in parentheses.

```python
>>> sqrt(2)
1.4142135623730951

>>> sin(2pi/3), sqrt(3)/2
0.8660254037844387, 0.8660254037844386

# Multiple arguments example
>>> log(3^9, 3)
9
```

### Implicit calls

Functions can be called *implicitly*, meaning the parentheses may be omitted. 
In this case, the function call has precedence between multiplication and 
addition.

```python
>>> cos2pi
1

# This example is equivalent to sin(2/3π^2) + 10
>>> sin2/3pi^2+10
10.292223461899434
```

Some functions have difference precedence, for example `fact(n)`, or factorial, 
has the same precedence as exponentiation. Some informational functions, such 
as `help()`, have very low precedence.

```python
>>> fact3*2
12

>>> fact(3*2)
720

>>> help 1+1
help: a + b
Addition operator
```

### Zero-argument calls

Functions that take zero arguments can be called implicitly or explicitly. They 
cannot, however, be given any arguments.

```python
# Assume a zero-arg function f() = 3

# Explicit call
>>> f()
3

# Implicit call
>>> f
3

# An explicit call with arguments is instead interpreted as an 
# implicit call followed by multiplication. This example is 
# equivalent to f()(7)
>>> f(7)
21
```

## Creating functions
Define a custom function by writing the function's signature, an equals sign, 
then the function body.

```python
f = calc.evaluate(ctx, 'f(x) = x^2')
print(f)
> f(x) = x^2
```

Defining a function using `calc.evaluate()` returns a `DeclaredFunction` object, 
which can be called like a regular python function.
```python
print( f(4) )
> 16
```

A `DeclaredFunction` can be added to the context, then it can be used in future 
expressions.
```python
ctx.add(f)
print( calc.evaluate(ctx, 'f(3+1)') )
> 16
```

Functions can be defined taking any number of arguments.
```python
# Example with multiple arguments
f = calc.evaluate(ctx, 'f(x, y) = x^3 * y^2')
print( f(3, 2) )
> 108

# Example with zero arguments
f = calc.evaluate(ctx, 'f() = round(100rand())')
print( f(), f(), f(), f(), f() )
> 46 95 79 21 64
```

## Creating variables
Defining a variable is the same as defining a function, but without the 
parentheses, since a variable doesn't take any arguments.

When defining a variable using `calc.evaluate()`, the returned value is a 
`DeclaredFunction` object, just as with a function, **not** the value of the 
variable itself. To get the variable's value, call the returned object.

```python
foo = calc.evaluate(ctx, 'foo = 7pi/3')
print(foo)
> foo = 7.330382858376184

print( foo() )
> 7.330382858376184

# Add the variable to the context to use in expressions
ctx.add(foo)
print( calc.evaluate(ctx, 'foo*3/7') )
> 3.141592653589793

# Unlike zero-arg functions, variables save their value
foo = calc.evaluate(ctx, 'foo = round(100rand())')
print( foo(), foo(), foo(), foo(), foo() )
> 78 78 78 78 78

```

Using `calc.console()`, defined functions and variables will be automatically 
added to the context.
```python
>>> n = 1+2+3
Added n to context.
n = 6

>>> f(x) = ln(x)^2
Added f(x) to context.
f(x) = ln(x)^2

>>> f(n)
3.210401995568401
```

## Function references
When a function is in an expression but is not called, a **reference** to the 
function itself is returned.

```python
f = calc.evaluate(ctx, 'sin')
print(f)
> sin(θ)
```

The returned value is a `FunctionDefinition` object, which can be called to 
evaluate the function.

```python
f = calc.evaluate(ctx, 'sqrt')
print( f(2) )
> 1.4142135623730951
```

`DeclaredFunction`, as seen earlier by defining custom functions, inherits from 
`FunctionDefinition`, and thus is also a function reference. In other words,
function references can be created by referring to an existing function by name,
or by creating a new function.

### Functions as arguments
Some functions take other functions as arguments. Integrals are an example. For
these functions, pass in a function reference as the argument.

```python
# Example of referencing a function by name and passing 
# it as an argument. This expression gives the definite 
# integral of sin(x)dx from 0 to π.
>>> int(sin, 0, pi)
2

# Example of using a function declaration as an argument
>>> int(f(t)=2t, 0, 5)
25
```

## Recursion

When creating a function, the function itself will be available to use in the 
body of the function, enabling recursion.

```python
# Example of a recursive factorial function. This uses the
# if function to check the base case, and recursively calls
# factorial(n-1).
factorial(n) = if(n<1, 1, n * factorial(n-1))

# Here's a more advanced example of a function that solves
# the Towers of Hanoi puzzle.
hanoi(n, a, b) =
if (
    n < 2,
    v(a, b),
    ( hanoi(n-1, a, 6-(a+b)), v(a, b), hanoi(n-1, 6-(a+b), b) )
)
```

## Advanced function creation
There are three advanced types of arguments you can create when defining a 
function: **function arguments**, **optional arguments**, and 
**star arguments**.

### Function arguments
If your function takes another function as an argument, add a pair of 
parentheses `()` to the name of the argument to signify that the argument is a
function.

```python
>>> call(f(), x) = f(x)
Added call(f(), x) to context.
call(f(), x) = f(x)

>>> call(sqrt, 2)
1.4142135623730951
```

When defining a function argument `f`, you do not need to specify how many 
arguments `f` takes, or any of their names. You only need to signify that `f`
is a function by adding the parentheses `f()`.

### Optional arguments
Optional arguments do not need to be passed into the function at all. Mark an 
argument as optional by adding a `?` to its name. If a value is not given for 
that argument, it will receive `None`.

```python
>>> f(b?) = b
Added f(b?) to context.
f(b?) = b

>>> f(3)
3

>>> f()
None
```

The [default context](#the-default-context) defines a variable `_` representing
`None`, as well the `?` operator, which are useful when using optional 
arguments. The `?` operator returns the left-hand side if it is not `None`, and
the right-hand side otherwise.

```python
# This function will return x*2 if x is given, and 3*2 if
# x is not given.
>>> f(x?) = (x ? 3) * 2
Added f(x?) to context.
f(x?) = (x?3)*2

>>> f(8)
16

>>> f()
6
```

`_` can be used to leave an optional argument blank explicitly. For example,
for a function `f(a?, b?)`, you can pass a value for `b` but not `a` by
calling `f(_, b)`.

Optional arguments must be defined after required arguments, and before 
[star arguments](#star-arguments).

### Star arguments
A star argument represents a collection of any number of arguments (this works
the same as star arguments in python). Define a star argument by adding an 
asterisk `*` in front of the argument name.

```python
>>> f(x, *star) = (star, x)
Added f(x, *star) to context.
f(x, *star) = (star, x)

>>> f(1)
1

>>> f(1, 2, 3, 4, 5)
2, 3, 4, 5, 1
```

The argument called `star` in this example will be a collection of all the 
values passed into `f`, except `x`.

A function can only have one star argument, and it must be the last argument.

### Combining argument types
An argument `x` can combine special argument types in two ways:

- `x()?` An optional function argument. The question mark must come after the 
parentheses.
- `*x()` Any number of function arguments.

Adding an optional modifier to a star argument has no effect. `*x?` is 
equivalent to `*x`.

## Lists

The concat operator (`,`) can be used to create lists. These act similarly to
lists in python, so they can be used with addition and multiplication.

```python
>>> 1, 2, 3
1, 2, 3

>>> ans + (4, 5)
1, 2, 3, 4, 5

>>> (1,2) * 3
1, 2, 1, 2, 1, 2

# Using concat on a list will combine everything into
# one list
>>> (1, 2), 3, (4, 5)
1, 2, 3, 4, 5
```

The concat operator has the lowest possible precedence, so make sure to use
parentheses when appropriate

```python
# Notice that x was not added to the context. This is 
# because concat has lower precedence than declaring
# a function/variable, so this expression is parsed
# as "(x = 1), 2".
>>> x = 1, 2
x = 1, 2

# Notice the parentheses this time
>>> x = (1, 2)
Added x to context.
x = (1, 2)
```

### Spread operator

The spread operator (`*x`) is a unary operator that spreads a list into separate
arguments to a function.

```python
>>> f(a, b, c) = a+b+c
Added f(a, b, c) to context.
f(a, b, c) = a+b+c

>>> x = (1, 2, 3)
Added x to context.
x = (1, 2, 3)

# x is treated as a single argument so we get an 
# error here
>>> f(x)
TypeError: f(a, b, c) expected 3 arguments, got 1

# We can use the spread operator to spread x into 
# 3 separate arguments
>>> f(*x)
6
```

## Vectors & matrices

The `v()` and `mat()` functions in the [default context](#the-default-context) create vectors 
and matrices respectively. Vectors are similar to lists, however they have 
different behavior with addition and multiplication:

```python
>>> v(1, 2)
<1, 2>

>>> v(1, 2) + v(3, 4)
<4, 6>

>>> v(1, 2) * 5
<5, 10>
```

`mat()` can be passed in a set of row vectors, or the arguments can be lists:

```python
>>> mat(v(1,2), v(3, 4))
[<1, 2>, <3, 4>]

>>> M = mat((1, 2, 3), (4, 5, 6), (7, 8, 9))
Added M to context.
M = [<1, 2, 3>, <4, 5, 6>, <7, 8, 9>]

# printmat() can be used to print a pretty version
# of a matrix
>>> printmat(M)
⎡  1  2  3  ⎤
⎢  4  5  6  ⎥
⎣  7  8  9  ⎦
```

Matrices and vectors support matrix multiplication and matrix-vector 
multiplication.


# Graphing
1-dimensional functions (1 input and 1 output) can be graphed using 
`calc.graph()`, or using the `graph()` function in the [default context](#the-default-context). 
The returned value is a matplotlib Figure. Matplotlib must be installed to use 
graphing.

```python
fig = calc.graph(ctx, 'f(x) = 1/25(x-8)(x-2)(x+8)')
fig.show()
```
![](https://i.imgur.com/VG5PQkp.png)

`graph()` accepts the following arguments:
1. `f()` A [reference](#function-references) to the function to be graphed. The 
function must take 1 number input and output 1 number. If the function returns 
something other than a number or throws an error, that point will not be plotted 
(eg. log, sqrt).
2. `xlow?` Lower bound for the x-axis. Default is -10.
3. `xhigh?` Upper bound for the x-axis. Default is 10.
4. `ylow?` Lower bound for the y-axis. Default is automatic scaling.
5. `yhigh?` Upper bound for the y-axis. Default is automatic scaling.
6. `n?` Number of points to evaluate. Default is 1000.

Calls to `graph()` inside the console will automatically call `.show()` on the 
returned figure.


# LaTeX
Expressions can be converted to LaTeX using `calc.latex()` or the `latex()` 
function in the [default context](#the-default-context).

```python
print( calc.latex(ctx, '3/4pi') )
> \frac{3}{4} \pi

print( calc.latex(ctx, 'int(f(t)=3t/4+6, -5, 5)') )
> \int_{-5}^{5} {\frac{3 t}{4} + 6} \, dt

print( calc.latex(ctx, 'f(x) = 3x^2+6') )
> f\left(x\right) = 3 x^{2} + 6
```

Many functions and operators have special LaTeX syntax to match mathematical 
notation, such as integrals, exponents, and division, as seen above.

```python
>>> latex( int(f(t) = 3t^2/4 + 6, -1/2, 1/2) )
\int_{\frac{-1}{2}}^{\frac{1}{2}} {\frac{3 t^{2}}{4} + 6} \, dt
```

![](https://i.imgur.com/LF7n9Rm.png)

If a custom variable or a reference to a custom function is given, the LaTeX 
will have the full definition.

```python
>>> p = pi^2
Added p to context.
p = 9.869604401089358

>>> latex(p)
p = \pi^{2}

>>> f(x) = 1/sqrtx
Added f(x) to context.
f(x) = 1/sqrt(x)

>>> latex(f)
f\left(x\right) = \frac{1}{\sqrt{x}}
```


# Scopes & contexts
A **scope** is a set of defined names or symbols that can be used in 
expressions. These include binary operators, unary operators, functions, and 
variables. A **context** is a stack of scopes.

You can create an empty context with the `calc.Context` class, but it's 
probably more useful to create a [default context](#the-default-context) using 
`calc.create_default_context()`.

```python
ctx = calc.Context()
# or
ctx = calc.create_default_context()
```

Use `ctx.add()` to add a new entry into the scope at the top of the context 
stack. `ctx.push_scope()` and `ctx.pop_scope()` can be used to push and pop 
scopes off the stack.

```python
ctx.push_scope()
f = calc.evaluate(ctx, 'f(x) = x^2+2')
ctx.add(f)
...
ctx.pop_scope()
```

However, if you expect you'll need to pop a scope, you should use the context 
manager `ctx.with_scope()` instead of pushing and popping scopes directly. This 
will guarantee the scope gets popped even if an error is raised.

```python
with ctx.with_scope():
    f = calc.evaluate(ctx, 'f(x) = x^2+2')
    ctx.add(f)
    ...
```

### Definitions
Entries in a context are called **definitions**. `FunctionDefinition`, 
`VariableDefinition` and `DeclaredFunction` are examples of definitions. Those
three have a **definition type** of **identifier**, meaning they are 
variables/functions, and are referred to by name, like `pi`, `sin`, or 
`foo_bar_3`.

There are three definition types: **binary operator**, **unary operator**, and
**identifier**. Every definition in a scope is uniquely identified by its 
name (symbol, in the case of operators), and its type. Two definitions with the
same name and type cannot exist in the same scope. These types are defined in
the `calc.DefinitionType` enum.

Two definitions with the
same name can exist in the same scope if they have different types. For example,
the default context defines a _binary operator_ and a _unary operator_ with the 
same symbol `-` for subtraction and negation, respectively.

```python
# (-3) - (-5) - 2 - (---6)
>>> -3 - -5 - 2 - ---6
6
```

To retrieve a definition from a context, use `ctx.get()`. Given a name and type,
it will return the highest definition with that name and type on the context 
stack. The type can be omitted, in which case it will default to _identifier_.

```python
# Example of adding a definition for a variable x
# then retrieving it
x = VariableDefinition('x', 5)
ctx.add(x)
x2 = ctx.get('x')
print( x2 is x )
> True

# Example of retrieving definitions with different
# types
from calc import DefinitionType
print( ctx.get('-', DefinitionType.BINARY_OPERATOR) )
> a - b
print( ctx.get('-', DefinitionType.UNARY_OPERATOR) )
> -x
```

### Shadowing
If a definition in one scope has the same name and type as a definition in a 
scope lower on the context stack, the definition in the higher scope is 
**shadowing** the definition in the lower scope. In other words, you can push a
scope and give a new value to anything already in the context.

```python
ctx.push_scope()
ctx.add(VariableDefinition('x', 5))

# ctx.get('x') will return x's definition, so call 
# the definition to get the value of x
print( ctx.get('x')() )
> 5

# Push a new scope
with ctx.with_scope():
    # Shadow the value x = 5
    ctx.add(VariableDefinition('x', 10))
    print( ctx.get('x')() )
    > 10

# The scope with x = 10 has been popped, so x will 
# be back to 5
print( ctx.get('x')() )
> 5
```

Shadowing is how variables can get redefined without replacing each other.
Every time a function is called, a new scope is pushed and its arguments are
added to the context, so they shadow variables with the same name.

```python
>>> x = 5
Added x to context.
x = 5

>>> f(x) = x
Added f(x) to context.
f(x) = x

# The argument x shadows the variable x defined earlier
>>> f(10)
10

>>> x
5
```

### Global scope
The scope at the bottom of the context stack is the **global scope**. This is
where the [default context](#the-default-context) defines all of its stuff. By
default (see [Settings](#settings)), the global scope is guarded against
modification and shadowing.

#### Global scope modification
`ctx.add()` cannot add to the global scope. If the context stack has only the 
global scope in it, using `ctx.add()` will raise a `ContextError`. To add
definitions to the global scope, use `ctx.add_global()`.

```python
ctx = Context()
ctx.add(VariableDefinition('x', 5))
```
<pre><code style="color: #FF6B68">ContextError: Cannot modify global scope</code></pre>

`ctx.remove()` as well as the `del()` function in the default context similarly 
cannot modify the global scope.

#### Global scope shadowing
Definitions in the global scope cannot be shadowed by default. This prevents
replacing common functions or variables like `pi` or `ln`, which would be 
pretty weird if allowed.
```python
# Sorry engineers, this wont work.
>>> pi = 3
```
<pre><code style="color: #FF6B68">ContextError: Cannot shadow 'pi' from global scope</code></pre>

### Settings
Contexts have a `settings` attribute, which has various things you can change
that affect how expressions are parsed and evaluated.
```python
ctx.settings.rounding = 6
```

Each setting has a docstring that explains what it does. Available settings 
are:
- `rounding`
  - Determines how many decimal places to round answers to
  - Default: None
- `parse_unknown_identifiers`
  - Whether the parser should defer checking if an identifier is defined to 
    evaluation
  - Default: False
- `auto_close_parentheses`
  - Whether to add missing closing parentheses at the end of an expression
  - Default: False
- `allow_global_scope_shadowing`
  - Whether to allow shadowing definitions in the global scope
  - Default: True
- `save_function_outer_scope`
  - Whether functions should remember their outer scope (see 
  [Saved scopes](#saved-scopes))
  - Default: True
- `saved_scope_shadowing`
  - Whether saved scopes should shadow or be shadowed by existing outer scopes
  - Default: False

See the previously mentioned docstrings for more details about these.

### The default context
`calc.create_default_context()` creates a new Context with a global scope 
containing dozens of useful variables, functions, and operators to use in 
expressions.

The following is a list of everything the default context includes.

#### Constants
|   `π`   |  `pi`  |  `ϕ`  |  `phi`  |  `∞`  |
|:-------:|:------:|:-----:|:-------:|:-----:|
|  `inf`  |  `j`   |  `_`  |         |       |

#### Binary operators
| `,` | `\|` | `&` | `>` | `<` |
|:---:|:----:|:---:|:---:|:---:|
| `+` | `-`  | `?` | `*` | `/` |
| `%` | `^`  |     |     |     |

#### Unary operators
| `-` | `!` | `*` |
|:---:|:---:|:---:|

#### Basic functions
| `abs(x)`  | `rad(θ)` | `deg(θ)` | `round(x)` | `floor(x)` |
|:---------:|:--------:|:--------:|:----------:|:----------:|
| `ceil(x)` | `ans()`  |          |            |            |

#### Informational functions
|     `echo(expr)`     | `type(obj)` | `help(obj)`  | `tree(expr)` | `graph(f(), xlow?, xhigh?, ylow?, yhigh?, n?)` |
|:--------------------:|:-----------:|:------------:|:------------:|:----------------------------------------------:|
| `latex(expr, eval?)` | `del(obj)`  | `scope(f())` |              |                                                |

#### Logic & data structure functions
| `sum(*x)` | `len(*x)` |      `filter(f(), *x)`      | `map(f(), *x)` | `range(a, b?, step?)` |
|:---------:|:---------:|:---------------------------:|:--------------:|:---------------------:|
| `max(*x)` | `min(*x)` | `if(condition, if_t, if_f)` |   `set(*x)`    |                       |

#### Roots & complex functions
| `sqrt(x)` | `root(x, n)` | `hypot(x, y)` |
|:---------:|:------------:|:-------------:|

#### Trigonometric functions
| `sin(θ)` | `cos(θ)`  | `tan(θ)`  | `sec(θ)`  |   `csc(θ)`    |
|:--------:|:---------:|:---------:|:---------:|:-------------:|
| `cot(θ)` | `asin(θ)` | `acos(θ)` | `atan(θ)` | `atan2(x, y)` |

#### Hyperbolic functions
| `sinh(x)` | `cosh(x)` | `tanh(x)` |
|:---------:|:---------:|:---------:|

#### Exponential & logarithmic functions
| `exp(x)` | `ln(x)` | `log10(x)` | `log(x, b)` |
|:--------:|:-------:|:----------:|:-----------:|

#### Combinatorial & statistics functions
| `fact(n)` | `perm(n, k)`  | `choose(n, k)` | `binom(p, x, n)` | `fib(n)` |
|:---------:|:-------------:|:--------------:|:----------------:|:--------:|
| `rand()`  | `randr(a, b)` |   `avg(*x)`    |   `median(*x)`   |          |

#### Calculus
| `int(f(), a, b)` | `deriv(f(), x, n?)` |
|:----------------:|:-------------------:|

#### Vectors & matrices
|   `v(*x)`    |   `dot(v, w)`   |  `mag(v)`   |   `mag2(v)`   |  `norm(v)`   |
|:------------:|:---------------:|:-----------:|:-------------:|:------------:|
|  `zero(d)`   |  `mat(*rows)`   |   `I(n)`    |  `shape(M)`   | `mrow(M, r)` |
| `mcol(M, c)` | `mpos(M, r, c)` | `transp(M)` | `printmat(M)` |  `vi(v, i)`  |

#### Linear algebra
| `det(M)` | `rank(M)` | `nullsp(M)` | `rref(M)` | `lu(M)` |
|:--------:|:---------:|:-----------:|:---------:|:-------:|
| `svd(M)` |           |             |           |         |

#### Coordinate system conversion functions
|   `polar(x, y)`   |   `cart(r, θ)`    | `crtcyl(x, y, z)` | `crtsph(x, y, z)` | `cylcrt(ρ, ϕ, z)` |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| `cylsph(ρ, ϕ, z)` | `sphcrt(r, θ, ϕ)` | `sphcyl(r, θ, ϕ)` |                   |                   |

The default context has a `help()` function, which will provide a short 
description of whatever is passed into it. Use it to find out more about what
is available in the default context.

```
>>> help(pi)
help: π = 3.141592653589793
Ratio of a circle's circumference to its diameter

>>> help(rad)
help: rad(θ)
Converts `θ` in degrees to radians

>>> help(1+1)
help: a + b
Addition operator
```


# Python definitions
A `FunctionDefinition` instance defines the name, arguments, and implementation 
for a function. Use it to add your python functions to a context so they can be
used in an expression.

The `FunctionDefinition` constructor takes 3 required arguments:
- `name` The name of the function
- `args` Names of the arguments the function takes (should be an iterable of 
strings). This is the same syntax as defining functions inside an expression. 
See [creating functions](#creating-functions).
- `func` The function itself

```python
from calc import FunctionDefinition
ctx = calc.create_default_context()
ctx.push_scope()

def my_function(a, b, c):
    return a + b + c

ctx.add(
    # The args are an iterable of strings, so 'abc' is equivalent
    # to ['a', 'b', 'c']
    FunctionDefinition('myfun', 'abc', my_function)
)
print( calc.evaluate(ctx, 'myfun(5, 4, 3)') )
> 12

# Example with a function argument
def my_function2(f, a, b):
    return a * f(b)

ctx.add(
    # Use f() syntax to signify that f is a function
    FunctionDefinition('myfun2', ['f()', 'a', 'b'], my_function2)
)
print( calc.evaluate(ctx, 'myfun2(f(x) = x^2, 2, 3)') )
> 18
```

### Binary operators

Binary operators an unary operators work the same way, using 
`BinaryOperatorDefinition` and `UnaryOperatorDefinition` respectively.

The `BinaryOperatorDefinition` constructor takes 4 required arguments:
- `symbol` The operator's symbol. Currently only single characters are supported 
for symbols.
- `func` A function that implements the operator. This function should take 2
arguments, for the left and right operands.
- `precedence` This number determines the order of operations. Higher number 
means higher precedence. See 
[Precedence & Associativity](#precedence--associativity).
- `associativity` Can be either left-to-right or right-to-left, found in the 
`calc.Associativity` enum. Associativity determines the direction in which the 
operator is evaluated when multiple are strung together. See 
[Precedence & Associativity](#precedence--associativity).

```python
from calc import BinaryOperatorDefinition, Associativity

def bin_xor(a, b):
    return a ^ b

ctx.add(
    # A precedence of 4 is equivalent to addition
    BinaryOperatorDefinition('x', bin_xor, 4, Associativity.L_TO_R)
)
print( calc.evaluate(ctx, '123 x 456') )
> 435
```

### Unary operators

The `UnaryOperatorDefinition` constructor is the similar to binary operators,
with three differences:
- `func` should be a function that takes one argument, since a unary operator 
has one operand.
- `precedence` is optional. Default is 7 (equal to exponentiation).
- There is no `associativity` argument. Only left-hand unary operators are 
currently supported, so all unary operators are right-to-left associative.

```python
from calc import UnaryOperatorDefinition

def bin_flip(x):
    return ~x

ctx.add(
    # A precedence of 4 is equivalent to addition
    UnaryOperatorDefinition('~', bin_flip)
)
print( calc.evaluate(ctx, '~25') )
> -26
```

### Variables

`VariableDefinition` takes only two required arguments, `name` and `value`. 
These two are fairly self-explanatory.

```python
from calc import VariableDefinition

ctx.add(
    VariableDefinition('x', 1.23456)
)
print( calc.evaluate(ctx, '2x') )
> 2.46912
```

### Other arguments

These arguments are optional.

- `display_name` (for functions and variables) This is a secondary name that 
will be used when the definition is printed. `pi` uses this, for example, to
print `π`.
- `latex` A function that converts the [syntax tree](#syntax-trees) into LaTeX.
For variables, this is a string instead of a function.
- `help_text` Text shown when using `help()` on the definition. If no help text
is provided, the docstring associated with the definition's `func` will be used.
- `manual_eval` If enabled, arguments will not be evaluated before being passed
into the definition's `func`, and instead the [nodes](#syntax-trees) will be 
passed directly. Use `calc.evaluate(node)` to evaluate an input node.

More details about these can be found in the docstring under 
`calc.FunctionDefinition.__init__`.


# Evaluation details

### Precedence & Associativity

[Precedence](https://en.wikipedia.org/wiki/Order_of_operations) determines the 
order operators are evaluated when there are no parentheses. Operators with 
high precedence are evaluated before operators with low precedence.

`1 + 2 * 3` is evaluated as `1 + (2 * 3)` because multiplication has higher 
precedence than addition. Higher number means higher precedence; this is 
different from most programming language standards such as C and C++, which use 
use low number to mean high precedence.

[Associativity](https://en.wikipedia.org/wiki/Operator_associativity)
determines the direction of evaluation when multiple operators of the same 
precedence are strung together without parentheses.

Most binary operators are left-to-right associative, meaning they are evaluated
by reading left to right. For example, `1 / 2 / 3` is equivalent to 
`(1 / 2) / 3`. Exponentiation is an example of a right-to-left associative 
operator; `1 ^ 2 ^ 3` is equivalent to `1 ^ (2 ^ 3)`.

If two operators have the same precedence and different associativites, the 
operator with right-to-left associativity will take priority. This is an 
arbitrary decision.
[C](https://en.cppreference.com/w/c/language/operator_precedence) and 
[C++](https://en.cppreference.com/w/cpp/language/operator_precedence) for 
example, don't have any operators with the same precedence and different 
associativites, and neither does the [default context](#the-default-context).

The binary operators in the default context have the following precedence and
associativity:

| Symbol | Description    | Precedence | Associativity |
|:------:|----------------|------------|---------------|
|  `,`   | Concat         | 0          | L-R           |
|  `\|`  | Bitwise OR     | 1          | L-R           |
|  `&`   | Bitwise AND    | 2          | L-R           |
|  `>`   | Greater than   | 3          | L-R           |
|  `<`   | Less than      | 3          | L-R           |
|  `+`   | Addition       | 4          | L-R           |
|  `-`   | Subtraction    | 4          | L-R           |
|  `?`   | Coalesce       | 4          | L-R           |
|  `*`   | Multiplication | 6          | L-R           |
|  `/`   | Division       | 6          | L-R           |
|  `%`   | Remainder      | 6          | L-R           |
|  `^`   | Exponentiation | 7          | R-L           |

Regular functions have a default precedence of 5. Unary operators and functions
are always right-to-left associative.

### Syntax trees

Evaluating an expression has two steps: parsing and evaluating. Parsing an 
expression yields a **syntax tree**, a data structure that organizes every 
part of the expression (numbers, operators, functions, etc) in a heirarchy. 
By parsing an expression into a syntax tree, the expression can be efficiently 
evaluated many times with new inputs.

Every node in a syntax tree inherits from `calc.parser.Node`. Nodes can be 
passed back into `calc.evaluate()` or `calc.latex()`.

Use `calc.tree()` or the `tree()` function in the default context to see a
visualization for the syntax tree for a given expression.

```python
>>> tree( 2pi^2+3 )
Expression 2π^2+3
0 BinaryOperator(+)
├── 0 BinaryOperator(*)
│   ├── 0 Number(2)
│   └── 1 BinaryOperator(^)
│       ├── 0 Variable(pi)
│       └── 1 Number(2)
└── 1 Number(3)

# Each declaration will be printed as a separate tree
>>> tree( int(f(x)=2x^2, -1/2, 1/2) )
Declaration f(x) = 2x^2
0 BinaryOperator(*)
├── 0 Number(2)
└── 1 BinaryOperator(^)
    ├── 0 Variable(x)
    └── 1 Number(2)

Expression int(f(x) = 2x^2, -1/2, 1/2)
0 FunctionCall
├── 0 Function(int)
├── 1 Declaration(f(x))
├── 2 BinaryOperator(/)
│   ├── 0 UnaryOperator(-)
│   │   └── 0 Number(1)
│   └── 1 Number(2)
└── 3 BinaryOperator(/)
    ├── 0 Number(1)
    └── 1 Number(2)
```

### Saved scopes

When a [new function is declared](#creating-functions), a copy of its outer 
scope is saved inside the resulting `DeclaredFunction` object. This saved scope
will contain the identifiers used by the function body. Definitions from the
global scope won't be saved.

For example, consider a function `f(x) = xyz+pi`. Its _required identifiers_ are
`y` and `z`. Assuming those two variables are in the context when the function 
was created, they will be saved into `f`'s saved scope. `x` is not a required 
identifier because it is part of `f`'s arguments, and will not be saved. `pi` 
will also not be saved because it is in the global scope.

When a function that has a saved scope is called, its scope is inserted into 
the context stack.

#### Usage example

```python
>>> y = 5
Added y to context.
y = 5

# y = 5 will be stored in f's saved scope
>>> f(x) = xy
Added f(x) to context.
f(x) = xy

# Delete y out of the context
>>> del(y)
1

# y = 5 is still saved inside f, so f(2) is 2*5,
# even though y is not defined in the context.
>>> f(2)
10
```

You can inspect the contents of a function's saved scope using the `scope(f)`
function in the default context.

```python
>>> scope(f)
Outer scope of f(x) {
    y = 5
}
```

#### Saved scope settings

There are two [context settings](#settings) that change the behavior of saved
scopes. If `save_function_outer_scope` is False, the saved scope feature will
be disabled. In the above usage example, `f(2)` would raise a `ContextError`
since `y` will not be defined:

```python
# ctx.settings.save_function_outer_scope is 
# disabled in this example

>>> y = 5
Added y to context.
y = 5

>>> f(x) = xy
Added f(x) to context.
f(x) = xy

>>> del(y)
1

# y is no longer defined, so this will throw 
# an error
>>> f(2)
ContextError: 'y' is undefined.
```

By default, identifiers that are in a saved scope will be [shadowed](#shadowing)
by identifiers in the context. This allows you to change values of variables to
change the behavior of functions.

```python
>>> y = 5
Added y to context.
y = 5

# y = 5 will be stored in f's saved scope
>>> f(x) = xy
Added f(x) to context.
f(x) = xy

# Change the value of y
>>> y = 10
Added y to context.
y = 10

# The new value y = 10 will shadow the saved 
# value of y = 5, so f(2) = 2*10
>>> f(2)
20
```

If `saved_scope_shadowing` is True, then the saved scope will shadow identifiers
in the context, instead of being shadowed by them.

```python
# ctx.settings.saved_scope_shadowing is enabled 
# in this example

>>> y = 5
Added y to context.
y = 5

# y = 5 will be stored in f's saved scope
>>> f(x) = xy
Added f(x) to context.
f(x) = xy

# Change the value of y
>>> y = 10
Added y to context.
y = 10

# The saved value y = 5 will shadow the new 
# value of y = 10, so f(2) = 2*5
>>> f(2)
10
```

#### Returning functions

One of the main purposes of saving scopes is so that a function can be returned 
by another function, and it will remember arguments passed into the outer 
function.

For example, consider the function `f(x) = g(y) = xy`. When `f(3)` is called,
a new function `g(y)` is created, and the value `x = 3` will be saved into 
`g`'s saved scope.

```python
>>> f(x) = g(y) = xy
Added f(x) to context
f(x) = g(y) = xy

>>> f(3)
Added g(y) to context.
g(y) = xy

>>> scope(g)
Outer scope of g(y) {
    x = 3
}

>>> g(5)
15
```

# Saving to JSON
Contexts (minus the root context) can be saved to json files using `calc.dump_contexts()` and loaded using `calc.load_contexts()`.

```python
ctx = calc.create_default_context()
ctx.push_context()

ctx.add(calc.evaluate(ctx, 'f(x) = 3x^2 + 4'))
ctx.add(calc.evaluate(ctx, 'g(x) = 3/2x^3 + 4x - 1'))
ctx.push_context()
ctx.set('foo', 75.24623)

# dump_contexts() returns a list of contexts that can be
# converted to json
data = calc.dump_contexts(ctx)
with open('saved_math.json', 'w') as f:
    f.write(json.dumps(data))
```
```json
[{"f": "f(x) = 3*x^2+4", "g": "g(x) = 3/2*x^3+4*x-1"}, {"foo": "75.24623"}]
```

```python
ctx = calc.create_default_context()
# note that a new context is not pushed before loading

with open('saved_math.json') as f:
    data = json.loads(f.read())
calc.load_contexts(ctx, data)
calc.evaluate(ctx, 'f(foo), g(foo)')
> (16989.9853876387, 639365.6665474502)
```
