from .context import Context
from .definitions import is_identifier, Associativity, DefinitionType, Definition, BinaryOperatorDefinition, \
    UnaryOperatorDefinition, VariableDefinition, DeclaredFunction, ArgumentError, spread, replace_latex_symbols


def parse(ctx:Context, expr:str, start:int=0, end:int=None, allow_empty=False):
    """
    Parse an expression into a syntax tree.

    :param ctx: Scope object defining operators, identifiers, etc.
    :param expr: Expression string to parse
    :param start: Index of expr to start at (default 0)
    :param end: Index of expr to end at (default len(expr))
    :param allow_empty: If True, empty expressions return an empty ListNode. Otherwise, raise ExpressionSyntaxError.
    :return: ListNode object at the root of the syntax tree.
    """
    if end is None:
        end = len(expr)

    i = start
    root = ListNode()
    node = root
    expected = [Parenthesis, UnaryOperator, Number, Declaration, Identifier, EndOfExpression]
    prev = StartOfExpression

    while i < end:
        for cls in expected:
            # Parse the next token
            node, i, next_expected = cls.parse(ctx, node, i, expr, start, end)
            if next_expected is not None:
                # Token sucessfully parsed
                expected = next_expected
                prev = cls
                break
        else:
            # Expected types not found
            expected = ', '.join(map(lambda cls: cls.__name__, expected))
            raise ExpressionSyntaxError(
                'Expected one of [{}] after {}'
                .format(expected, prev.__name__), expr, i
            )

    # Check for unexpected end of expression
    if EndOfExpression not in expected:
        expected = ', '.join(map(lambda cls: cls.__name__, expected))
        raise ExpressionSyntaxError(
            'Unexpected end of expression. Expected one of [{}] after {}'
            .format(expected, prev.__name__), expr, i
        )

    # Empty expression
    if len(root.children) == 0 and not allow_empty:
        expected = ', '.join(map(lambda cls: cls.__name__, expected))
        raise ExpressionSyntaxError(
            'Expression is empty'
            .format(expected), expr, i
        )

    return root


class ExpressionSyntaxError(Exception):
    def __init__(self, msg, expr, i, length=1):
        super().__init__(msg)
        self.i = i
        self.length = max(1, length)
        self.expr = expr

    def __str__(self):
        return super().__str__() + '\n{}\n{}{}'.format(
            self.expr,
            ' ' * self.i,
            '^' * self.length
        )


class Token:
    """
    A token is a single piece of an expression, such as a number, operator, function, etc. Inherit from this class and
    implement ``parse()`` and ``next_expected()`` to use as a token. If the token should live on the final syntax tree,
    inherit from Node. The next_expected function can be either a class method or instance method.
    """
    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        """
        Parse one token of this type. If a node is sucessfully parsed, update the syntax tree accordingly. Returns the
        updated current working node, the updated expression index, and a list of next expected token types.

        :param ctx: The context
        :param node: The current working node, i.e. the node representing the most recently parsed token
        :param i: Curent character index of expr
        :param expr: The expression string
        :param start: The start index of expr (in case only part of the string is currently being parsed)
        :param end: The end index of expr
        :return: Node, int, List[Node class]
        """
        raise NotImplemented


class Node(Token):
    """
    Nodes are tokens that live as a node on syntax tree. If a token creates some other node(s) while parsing, inherit
    from Token instead of Node.
    """
    precedence = -1
    associativity = Associativity.L_TO_R

    def __init__(self, precedence=None, associativity=None):
        self.children = []
        self.parent = None
        if precedence is not None:
            self.precedence = precedence
        if associativity is not None:
            self.associativity = associativity

    def next_expected(self):
        """ Returns a list of Tokens expected after this token. """
        raise NotImplemented

    def evaluate(self, ctx:Context):
        """ Evaluate this node """
        raise NotImplemented

    def _eval_children(self, ctx:Context, definition:Definition):
        """ Evaluate each child node and yield the results """
        # Yield children nodes instead of evaluated results if the definition is manual_eval
        if definition.manual_eval:
            for child in self.children:
                yield child
            return

        for child in self.children:
            result = child.evaluate(ctx)
            # Special case for spread operator: yield all items in the spread instead of
            # the spread object itself
            if isinstance(result, spread):
                for item in result:
                    yield item
            else:
                yield result

    def __str__(self):
        raise NotImplemented

    def __repr__(self):
        raise NotImplemented

    def latex(self, ctx):
        """ Convert this node to a LaTeX string """
        raise NotImplemented

    def _tree_tag(self):
        """ Returns a string used as the tag for this node when added to a treelib.Tree """
        raise NotImplemented

    def add_child(self, node):
        """ Insert a node below self """
        node.parent = self
        self.children.append(node)

    def insert_parent(self, node):
        """ Insert a node above self """
        if self.parent:
            i = self.parent.children.index(self)
            self.parent.children[i] = node
        node.parent = self.parent
        node.add_child(self)

    def higher_precedence(self, other):
        """
        Returns True if this node's precedence is greater than `other`, in other words that self should be evaluated
        before `other`.
        """
        # If precedence is not equal, the higher one should be evaluated first
        if self.precedence != other.precedence:
            return self.precedence > other.precedence

        # If the precedence and associativities are the same, self should be first if it is left-to-right associative
        if self.associativity == other.associativity:
            return self.associativity == Associativity.L_TO_R

        # If self is left-associative and other is right-associative, evaluate other before self
        # If self is right-associative and other is left-associative, evaluate self before other
        return self.associativity == Associativity.R_TO_L

    def is_left_parenthesized(self, child):
        """
        Returns true if `child` (to the left of the parent) should be parenthesized. A node is parenthesized if it is a
        list or operator and has lower precedence than its parent.
        """
        if not isinstance(child, (ListNode, BinaryOperator, UnaryOperator)):
            return False
        return not child.higher_precedence(self)

    def is_right_parenthesized(self, child):
        """
        Returns true if `child` (to the right of the parent) should be parenthesized. A node is parenthesized if it is
        a list or operator and has lower precedence than its parent.
        """
        if not isinstance(child, (ListNode, BinaryOperator, UnaryOperator)):
            return False
        return self.higher_precedence(child)

    def propagate_precedence(self, binop):
        """ Propagate up the tree and return the first node with lower precedence than the given operator. """
        # If associativity is left-to-right, keep propagating up when precedence is equal.
        node = self
        while node.parent and node.parent.higher_precedence(binop):
            node = node.parent
        return node

    def leftmost_leaf(self):
        """
        Propagate down the tree and returns the leftmost leaf of this node. The leftmost leaf is the node immediately
        to the right of this node in the original expression.
        """
        if len(self.children) == 0 or self.associativity == Associativity.L_TO_R:
            # If the node is left-to-right associative, then it is considered to be to the left of its child nodes
            return self

        return self.children[0].leftmost_leaf()

    def rightmost_leaf(self):
        """ Propagate down the tree and returns the rightmost leaf of this node """
        if len(self.children) == 0 or self.associativity == Associativity.R_TO_L:
            # If the node is right-to-left associative, then it is considered to be to the right of its child nodes
            return self

        return self.children[-1].rightmost_leaf()

    def add_to_tree(self, tree, num):
        """ Add this syntax tree to a treelib.Tree. `num` is added to the tag to keep children in order. """
        tree.create_node(
            str(num) + ' ' + self._tree_tag(),
            id(self),
            id(self.parent) if self.parent else None
        )
        for i, node in enumerate(self.children):
            node.add_to_tree(tree, i)


class ListNode(Node):
    def evaluate(self, ctx:Context):
        if len(self.children) == 0:
            return None
        if len(self.children) == 1:
            return self.children[0].evaluate(ctx)

        # Evaluate children and concatenate them together
        concat = ctx.get(',', DefinitionType.BINARY_OPERATOR)
        result = []
        for child in self._eval_children(ctx, concat):
            result = concat.func(result, child)
        return result

    def __str__(self):
        return ', '.join(map(str, self.children))

    def __repr__(self):
        return '<{} children={}>'.format(type(self).__name__, len(self.children))

    def latex(self, ctx):
        return ',\, '.join(node.latex(ctx) for node in self.children)

    def _tree_tag(self):
        return '{}()'.format(type(self).__name__)


class EndOfExpression(Token):
    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        return node, i, None

class StartOfExpression(Token):
    pass


class BinaryOperator(Node):
    def __init__(self, op:BinaryOperatorDefinition):
        super().__init__(op.precedence, op.associativity)
        self.symbol = op.name

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        # Get the operator definition from the context
        op = ctx.get(expr[i], DefinitionType.BINARY_OPERATOR, default=None)
        if op is None:
            return node, i, None

        binop = cls(op)
        node = node.propagate_precedence(binop)

        # Special case for concatenation operator. Instead of adding it as a node, continue adding children to the
        # parent node
        if binop.symbol == ',':
            return node.parent, i+1, binop.next_expected()

        node.insert_parent(binop)
        return binop, i+1, binop.next_expected()

    def next_expected(self):
        return [Parenthesis, UnaryOperator, Number, Declaration, Identifier]

    def evaluate(self, ctx):
        op = ctx.get(self.symbol, DefinitionType.BINARY_OPERATOR)
        return op(*self._eval_children(ctx, op))

    def __str__(self):
        left = self.children[0]
        right = self.children[1]
        parens_left = self.is_left_parenthesized(left)
        parens_right = self.is_right_parenthesized(right)
        implicit = ImplicitMultiplication.is_implicit(self, left, right)

        left = str(left)
        right = str(right)

        if parens_left:
            left = '(' + left + ')'
        if parens_right:
            right = '(' + right + ')'

        if implicit:
            return left + right
        else:
            return left + self.symbol + right

    def __repr__(self):
        return '<{} symbol={}, children={}>'.format(
            type(self).__name__,
            repr(self.symbol),
            len(self.children)
        )

    def latex(self, ctx):
        definition = ctx.get(self.symbol, DefinitionType.BINARY_OPERATOR)

        left = self.children[0]
        right = self.children[1]
        parens_left = self.is_left_parenthesized(left)
        parens_right = self.is_right_parenthesized(right)
        implicit = ImplicitMultiplication.is_implicit(self, left, right)

        if definition.latex_func:
            return definition.latex_func(
                ctx, self, left, right,
                parens_left, parens_right, implicit
            )

        left = left.latex(ctx)
        right = right.latex(ctx)

        if parens_left:
            left = r'\left( ' + left + r' \right)'
        if parens_right:
            right = r'\left( ' + right + r' \right)'

        if implicit:
            return left + ' ' + right
        else:
            return '{} {} {}'.format(left, replace_latex_symbols(definition.name), right)

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.symbol)


class UnaryOperator(Node):
    def __init__(self, op:UnaryOperatorDefinition):
        super().__init__(op.precedence, op.associativity)
        self.symbol = op.name

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        op = ctx.get(expr[i], DefinitionType.UNARY_OPERATOR, default=None)
        if not isinstance(op, UnaryOperatorDefinition):
            return node, i, None

        unop = cls(op)
        node.add_child(unop)
        return unop, i+1, unop.next_expected()

    def next_expected(self):
        return [Parenthesis, UnaryOperator, Number, Declaration, Identifier]

    def evaluate(self, ctx:Context):
        op = ctx.get(self.symbol, DefinitionType.UNARY_OPERATOR)
        return op(*self._eval_children(ctx, op))

    def __str__(self):
        right = self.children[0]

        if self.is_right_parenthesized(right):
            right = '(' + str(right) + ')'

        return self.symbol + str(right)

    def __repr__(self):
        return '<{} symbol={}, children={}>'.format(
            type(self).__name__,
            repr(self.symbol),
            len(self.children)
        )

    def latex(self, ctx):
        definition = ctx.get(self.symbol, DefinitionType.UNARY_OPERATOR)

        right = self.children[0]
        parens_right = self.is_right_parenthesized(right)

        if definition.latex_func:
            return definition.latex_func(ctx, self, right, parens_right)

        right = right.latex(ctx)
        if parens_right:
            right = r'\left(' + right + r'\right)'
        return replace_latex_symbols(definition.name) + right

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.symbol)


class Parenthesis(Token):
    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        if expr[i] != '(':
            return node, i, None

        # Search for matching close parenthesis
        close = cls.find_close(expr, i, end)

        # Parse expression inside the parentheses
        root:ListNode = parse(ctx, expr, start=i+1, end=close)

        if isinstance(node, (Function, Declaration)):
            # If the current node is a function, keep the working node on the function. Add all child nodes of the
            # parenthesized expression root as children of the function.
            for child in root.children:
                node.add_child(child)
        else:
            # Otherwise, move the working node to the new root of the parenthesized expression.
            if len(root.children) == 1:
                root = root.children[0]
            node.add_child(root)
            node = root

        return node, close + 1, cls.next_expected(node)

    @classmethod
    def next_expected(cls, root):
        if isinstance(root, (Identifier, Declaration)) and not root.is_const and not root.explicit:
            # If the parentheses contained a function then tokens after the parentheses should be treated as a function
            # call (Ex. "(sin)(3)" or "(f(x)=2x)(4)")
            return [BinaryOperator, FunctionCall, EndOfExpression]
        return [BinaryOperator, ImplicitMultiplication, EndOfExpression]

    @classmethod
    def find_close(cls, expr, i, end, err=True):
        """
        Locates close parenthesis. If not found, throw an ExpressionSyntaxError.
        All parentheses between `i` and `end` must be balanced in order to find the correct answer.
        Assumes that expr[i] is '('

        :param expr: Expression string
        :param i: Index of the opening parenthesis
        :param end: Index to stop searching
        :param err: If True, throw an error if ) wasn't found, otherwise return -1
        :return: Index of the matching close parenthesis
        """
        parens = 1
        j = i + 1
        while parens > 0 and j < end:
            ch = expr[j]
            if ch == '(':
                parens += 1
            elif ch == ')':
                parens -= 1
            j += 1

        if parens > 0:
            if err:
                raise ExpressionSyntaxError('No matching close parenthesis', expr, i)
            else:
                return -1

        return j - 1


class FunctionCall(Token):
    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        # Explicit empty call. Remove the () and continue.
        if expr[i:i+2] == '()':
            node.explicit = True
            return node, i+2, cls.next_expected(True)

        # Implicit call to a 0-arg function. Set as implicit but continue as explicit.
        elif node.n_args == 0:
            node.explicit = False
            return node, i, cls.next_expected(True)

        # Explicit non-empty call. Parse the inside of the parentheses.
        elif expr[i] == '(':
            node.explicit = True
            node, i, _ = Parenthesis.parse(ctx, node, i, expr, start, end)
            return node, i, cls.next_expected(True)

        # Implicit call.
        else:
            node.explicit = False
            return node, i, cls.next_expected(False)

    @classmethod
    def next_expected(cls, explicit):
        if explicit:
            return [BinaryOperator, ImplicitMultiplication, EndOfExpression]
        return [Number, Identifier, EndOfExpression]


class Number(Node):
    def __init__(self, n):
        super().__init__()
        self.value = n

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        n = ''
        decimal = -1
        is_int = True

        j = i
        while j < end:
            ch = expr[j]
            if ch.isdigit():
                # Eat a digit
                n += ch
                # Check if the number is still an integer
                # Any non-zero digit after a decimal means the number isn't an int
                if decimal != -1 and ch != '0':
                    is_int = False
            elif decimal == -1 and ch == '.':
                # Eat a decimal point
                # Ensure that only one decimal point exists
                n += ch
                decimal = j - i
            else:
                break
            j += 1

        # Not a valid number
        if n == '' or n == '.':
            return node, i, None

        # If the number is an int with a decimal point, there are excess zeroes, like "3.000"
        if is_int and decimal != -1:
            if decimal == 0:
                # edge case for ".0"
                n = 0
            else:
                # remove excess zeroes
                n = n[:decimal]

        # Convert string to float or int
        n = int(n) if is_int else float(n)

        num = cls(n)
        node.add_child(num)
        return num, j, num.next_expected()

    def next_expected(self):
        return [BinaryOperator, ImplicitMultiplication, EndOfExpression]

    def evaluate(self, ctx:Context):
        return self.value

    def __repr__(self):
        return '<{} n={}, children={}>'.format(
            type(self).__name__,
            repr(self.value),
            len(self.children)
        )

    def __str__(self):
        return str(self.value)

    def latex(self, ctx):
        val = ctx.round_result(self.value)
        return str(val)

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.value)


class Identifier(Node):
    def __init__(self, definition:Definition):
        super().__init__(definition.precedence, definition.associativity)
        self.name = definition.name
        self._display_name = getattr(definition, 'display_name', None)
        self.n_args = len(definition.args)
        self.is_const = definition.is_constant
        self.explicit = False

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        j = i
        string = ''
        definition = None

        # Find the longest continuous identifier that exists in the current context. Implicit multiplication & implicit
        # function calls can mean identifiers may be strung together (Ex. "sincosepi" = "sin(cos(e*pi))").
        #   `string` will contain the entire string "sincosepi", and j will be the end index of `string`.
        #   `definition` will be the definition of the first identifier in the context, in this example the definition
        #   for sin.
        # Parsing the rest will be delegated to the next token.
        while j < end:
            string += expr[j]
            if not is_identifier(string):
                break
            definition = ctx.get(string, default=definition)
            j += 1

        if j == i:
            # `string` is empty, in other words this token isn't an identifier.
            return node, i, None
        elif definition is None:
            if ctx.params.parse_unknown_identifiers:
                # If unknown identifiers is enabled, parse the whole string as a single variable, which will throw a
                # ContextError if it is still undefined at evaluation time.
                definition = VariableDefinition(expr[i:j], None)
            else:
                raise ExpressionSyntaxError("Undefined identifier '{}'".format(expr[i:j]), expr, i, j-i)

        if definition.is_constant:
            iden = Variable(definition)
        else:
            iden = Function(definition)

        node.add_child(iden)
        return iden, i + len(definition.name), iden.next_expected()

    def next_expected(self):
        raise NotImplemented

    def __str__(self):
        raise NotImplemented

    def __repr__(self):
        raise NotImplemented

    def latex(self, ctx):
        raise NotImplemented

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.name, self.n_args)


class Function(Identifier):
    def next_expected(self):
        return [BinaryOperator, FunctionCall, EndOfExpression]

    def evaluate(self, ctx:Context):
        definition:Definition = ctx.get(self.name, DefinitionType.IDENTIFIER)

        if len(self.children) == 0 and len(definition.args) > 0 and not self.explicit:
            # If the function takes arguments but was given none, and was not explicitly called, return the definition
            # itself.
            return definition

        # Evaluate arguments & pass it to the function
        inputs = list(self._eval_children(ctx, definition))
        return definition(*inputs)

    def __str__(self):
        name = self._display_name or self.name
        if len(self.children) == 0 and self.n_args > 0 and not self.explicit:
            return name

        def arg_str(child):
            # Add parentheses to the argument if it is a list
            if isinstance(child, ListNode):
                return '(' + str(child) + ')'
            return str(child)

        args = ', '.join(map(arg_str, self.children))
        return '{}({})'.format(name, args)

    def __repr__(self):
        return '<{} name={}, n_args={}, is_const={}, explicit={}, children={}>'.format(
            type(self).__name__,
            repr(self.name),
            self.n_args,
            self.is_const,
            self.explicit,
            len(self.children)
        )

    def latex(self, ctx):
        definition = ctx.get(self.name)
        n_inputs = len(self.children)
        name = replace_latex_symbols(self._display_name or self.name)

        if n_inputs == 0:
            if isinstance(self.parent, Function):
                # Function is being passed as an argument to another function. Use only the name.
                return name
            elif isinstance(definition, DeclaredFunction):
                # Use the full declaration.
                return Declaration(definition, definition.func).latex(ctx)
            else:
                # Use the function signature
                return replace_latex_symbols(definition.signature)

        definition.check_inputs(n_inputs)

        if definition.latex_func:
            return definition.latex_func(ctx, self, *self.children)

        return r'{}\left( {} \right)'.format(
            name,
            ', '.join(node.latex(ctx) for node in self.children)
        )


class Variable(Identifier):
    def next_expected(self):
        return [BinaryOperator, ImplicitMultiplication, EndOfExpression]

    def evaluate(self, ctx:Context):
        definition = ctx.get(self.name, DefinitionType.IDENTIFIER)
        return definition()

    def __str__(self):
        return self._display_name or self.name

    def __repr__(self):
        return '<{} name={}, children={}>'.format(
            type(self).__name__,
            repr(self.name),
            len(self.children)
        )

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.name)

    def latex(self, ctx):
        definition = ctx.get(self.name, default=None)

        name = replace_latex_symbols(
            getattr(definition, 'latex_name', None)
            or self._display_name
            or self.name
        )

        if definition and self.parent is None:
            if hasattr(definition.func, 'latex'):
                # Direct variable reference passed, use full declaration
                return '{} = {}'.format(name, definition.func.latex(ctx))

        return name


class ImplicitMultiplication(Token):
    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        op = ctx.get('*', DefinitionType.BINARY_OPERATOR, default=None)
        if not isinstance(op, BinaryOperatorDefinition):
            # No multiplication operator in the ctx.
            return node, i, None

        mul = BinaryOperator(op)
        node = node.propagate_precedence(mul)
        node.insert_parent(mul)
        return mul, i, cls.next_expected()

    @classmethod
    def next_expected(cls):
        return [Parenthesis, Identifier]

    @staticmethod
    def is_implicit(binop, left, right):
        """ Whether the expression ``left [binop] right`` can be implicit. """
        if not binop.symbol == '*':
            # Implicit operators are only supported for multiplication
            return False

        if isinstance(right, Number):
            # Implicit multiplication can never be followed by a number
            return False

        if isinstance(right, BinaryOperator) and binop.symbol == right.symbol:
            # Don't use implicit multiplication if the right is also a multiply
            return False

        if right.higher_precedence(binop) and isinstance(right.leftmost_leaf(), Number):
            # Don't use implicit multiplication if the right node has higher precedence
            # than multiply (means it won't be parenthesized) and the leftmost token is a
            # number.
            # Ex. 4(2^3) -> 4*2^3 but 4(x^2) -> 4x^2
            return False

        return True


class Declaration(Identifier):
    """ A declaration of a new identifier. Takes the form of foo(a,b,c)=... """
    precedence = 1

    def __init__(self, definition:DeclaredFunction, root):
        super().__init__(definition)
        self.definition:DeclaredFunction = definition
        self.root = root

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        # Index of equals sign
        equals = expr.find('=', i, end)

        if equals == -1:
            # Not a declaration.
            return node, i, None

        # Parse the signature
        name, args, is_const = cls.parse_signature(expr, i, equals)

        if name is None:
            # If signature parsing fails, this isn't a valid declaration. Probably means there's a declaration later on
            # and this is an identifier.
            return node, i, None

        # Create a function definition
        try:
            definition = DeclaredFunction(name, args, is_const)
        except ArgumentError as e:
            raise cls._signature_error(e, name, args, expr, i) from None

        # Push the function definition and its argument variables to the context and parse the remainder of the
        # expression
        with ctx.with_scope():
            ctx.add(definition)
            definition.add_args_to_context(ctx, None)

            end = cls.find_expression_end(ctx, expr, equals+1, end)
            root:ListNode = parse(ctx, expr, start=equals+1, end=end)
            if len(root.children) == 1:
                root = root.children[0]
                root.parent = None

            definition.func = root

        # Finally, add the new identifier to the tree
        decl = cls(definition, root)
        node.add_child(decl)
        return decl, end, decl.next_expected()

    @classmethod
    def parse_signature(cls, expr, i, end):
        """
        Parses a function signature from a string and returns name, args array.

        Returns (name, args, const).
        `args` is a list of strings, see FunctionDefinition(). `is_const` is whether this declaration should be
        evaluated once then saved, or re-evalate each time it is used. This is True if the signature takes no arguments
        and does not have empty parentheses. For example "x() = 3y" would change when y changes, whereas "x = 3y" would
        not.
        If the signature is invalid, returns (None, None, None)

        :param expr: Expression string
        :param i: Current index
        :param end: End index of the signature
        :return: identifier name, arg list, bool const
        """
        # index of opening parenthesis
        parens = expr.find('(', i, end)

        if parens == -1:
            # Variable declaration (0-arg function) without parentheses. Entire signature must be a single identifier.
            parens = end
            rparens = end - 1
            is_const = True
        else:
            # Locate closing parenthesis. Should be before the equals sign.
            rparens = Parenthesis.find_close(expr, parens, end, err=False)
            is_const = False

        if rparens == -1 or rparens != end - 1:
            # Closing parenthesis exists, but it is not immediately before the equals sign.
            return None, None, None

        # Name of declared identifier
        name = expr[i:parens]
        if not is_identifier(name):
            return None, None, None

        # Move start & end to capture the argument list
        i = parens + 1
        end = rparens

        # Parse argument list
        args = []
        if i < end:
            j = i
            while j < end:
                if expr[j] == ',':
                    args.append(expr[i:j])
                    i = j + 1
                j+=1
            # last arg
            args.append(expr[i:j])

        return name, args, is_const

    @staticmethod
    def _signature_error(e:ArgumentError, name, args, expr, i):
        # length of name plus one for the open parenthesis
        j = i + len(name) + 1
        for k in range(e.arg_i):
            # length of the arg plus one for the comma
            j += len(args[k]) + 1
        return ExpressionSyntaxError(str(e), expr, j, len(args[e.arg_i]))

    @classmethod
    def find_expression_end(cls, ctx, expr, start, end):
        """
        Given a declaration, determines the end of the expression after the equals sign. For example, in the expression
        "f(x)=3(3+2), 5, 7" the declaration for `f` could be "3+2" or "3+2, 5, 7" depending on precedence.

        Current implementation: Stop at first binary operator with greater precedence than Declaration.

        :param ctx: Context
        :param expr: Expression string
        :param start: Index of the expression start (after the equals sign)
        :param end: Index of the expression end
        :return: The new end index
        """
        i = start
        while i < end:
            ch = expr[i]
            if ch == '(':
                # Skip everything inside parentheses
                i = Parenthesis.find_close(expr, i, end)
            elif (ch, DefinitionType.BINARY_OPERATOR) in ctx:
                # Binary operator, check the precedence
                binop = ctx.get(ch, DefinitionType.BINARY_OPERATOR)
                if cls.precedence > binop.precedence:
                    break
            i += 1
        return i

    def next_expected(self):
        return [BinaryOperator, FunctionCall, EndOfExpression]

    def evaluate(self, ctx:Context):
        if len(self.children) == 0 and not self.explicit:
            # If the function takes arguments but was given none, and was not explicitly called, return the definition
            # itself.
            return self.definition

        # Evaluate arguments & pass it to the function
        inputs = self._eval_children(ctx, self.definition)
        return self.definition(*inputs)

    def __str__(self):
        if len(self.children) == 0:
            return str(self.definition)

        args = ', '.join(map(str, self.children))
        return '({})({})'.format(self.definition, args)

    def __repr__(self):
        return '<{} definition={}, children={}>'.format(
            type(self).__name__,
            repr(self.definition),
            len(self.children)
        )

    def latex(self, ctx):
        with ctx.with_scope():
            if not self.definition.name in ctx:
                ctx.add(self.definition)

            self.definition.add_args_to_context(ctx, None)

            signature = replace_latex_symbols(self.definition.signature)
            body = self.definition.func.latex(ctx)
            if isinstance(self.definition.func, ListNode):
                body = r'\left( ' + body + r' \right)'

            result = '{} = {}'.format(signature, body)

            if len(self.children) > 0:
                # Lambda call, Ex. "(f(x)=3x)(6)"
                result = r'\left( {} \right)\left( {} \right)'.format(
                    result,
                    ',\, '.join(node.latex(ctx) for node in self.children)
                )

        return result

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.definition.signature)

    def add_to_tree(self, tree, num):
        tree2 = type(tree)() # make a new tree
        self.root.add_to_tree(tree2, 0)
        print('Declaration', str(self.definition))
        tree2.show()
        super().add_to_tree(tree, num)
