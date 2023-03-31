from .context import Context
from .definitions import is_identifier, Associativity, DefinitionType, Definition, BinaryOperatorDefinition, \
    UnaryOperatorDefinition, DeclaredFunction, replace_latex_symbols


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
    expected = [Parenthesis, UnaryOperator, Number,
                Declaration, Identifier,
                EndOfExpression]

    while i < end:
        for cls in expected:
            # Parse the next token
            node, i, next_expected = cls.parse(ctx, node, i, expr, start, end)
            if next_expected is not None:
                # Token sucessfully parsed
                expected = next_expected
                break
        else:
            # Expected types not found
            expected = ', '.join(map(lambda cls: cls.__name__, expected))
            raise ExpressionSyntaxError(
                'Expected one of [{}]'
                .format(expected), expr, i
            )

    # Check for unexpected end of expression
    if EndOfExpression not in expected:
        expected = ', '.join(map(lambda cls: cls.__name__, expected))
        raise ExpressionSyntaxError(
            'Unexpected end of expression. Expected one of [{}]'
            .format(expected), expr, i
        )

    # Empty expression
    if len(root.children) == 0 and not allow_empty:
        expected = ', '.join(map(lambda cls: cls.__name__, expected))
        raise ExpressionSyntaxError(
            'Expression is empty'
            .format(expected), expr, i
        )

    # # Return the child node if there is only one
    # if len(root.children) == 1:
    #     root = root.children[0]
    #     root.parent = None

    return root


class ExpressionSyntaxError(Exception):
    def __init__(self, msg, expr, i, length=1):
        super().__init__(msg)
        self.i = i
        self.length = length
        self.expr = expr

    def __str__(self):
        return super().__str__() + '\n{}\n{}{}'.format(
            self.expr,
            ' ' * self.i,
            '^' * self.length
        )


class Token:
    """
    A token is a single piece of an expression, such as a number, operator, function, etc.
    Inherit from this class and implement parse() and next_expected() to use as a token. If the token
    should live on the final syntax tree, inherit from Node. The next_expected function can be either
    a class method or instance method.
    """

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        """
        Parse one token of this type. If a node is sucessfully parsed, update the syntax tree accordingly.
        Returns the updated current working node, the updated expression index, and a list of next expected token
        types.

        :param ctx: The context
        :param node: The current working node, i.e. the node representing the most recently parsed token
        :param i: Curent character index of expr
        :param expr: The expression string
        :param start: The start index of expr (in case only part of the string is currently being parsed)
        :param end: The end index of expr
        :return: SyntaxNode, int, List[Child class of SyntaxNode]
        """
        raise NotImplemented


class Node(Token):
    """
    Nodes are tokens that live as a node on syntax tree. If a token creates some other node(s)
    while parsing, inherit from Token instead of Node.
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
        raise NotImplemented

    def evaluate(self, ctx:Context):
        raise NotImplemented

    def _eval_children(self, ctx:Context, definition:Definition):
        """ Evaluate each child node and yield the results """
        if definition.manual_eval:
            yield ctx
            for child in self.children:
                yield child
        else:
            for child in self.children:
                yield child.evaluate(ctx)

    def __str__(self):
        raise NotImplemented

    def __repr__(self):
        raise NotImplemented

    def latex(self, ctx):
        raise NotImplemented

    def _tree_tag(self):
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
        """ Returns True if this node's precedence is greater than `other`, in other words
            that self should be evaluated before `other`. """
        # If precedence is not equal, the higher one should be evaluated first
        if self.precedence != other.precedence:
            return self.precedence > other.precedence

        # If the precedence and associativities are the same, a should be first if
        # it is left-to-right associative
        if self.associativity == other.associativity:
            return self.associativity == Associativity.L_TO_R

        # If a is left-associative and b is right-associative, evaluate b before a
        # If a is right-associative and b is left-associative, evaluate a before b
        return self.associativity == Associativity.R_TO_L

    def is_left_parenthesized(self, child):
        """ Returns true if `child` (to the left of the parent) should be parenthesized.
            A node is parenthesized if it is a list or operator and has lower precedence
            than its parent. """
        if not isinstance(child, (ListNode, BinaryOperator, UnaryOperator)):
            return False
        return not child.higher_precedence(self)

    def is_right_parenthesized(self, child):
        """ Returns true if `child` (to the right of the parent) should be parenthesized.
            A node is parenthesized if it is a list or operator and has lower precedence
            than its parent. """
        if not isinstance(child, (ListNode, BinaryOperator, UnaryOperator)):
            return False
        return self.higher_precedence(child)

    def propagate_precedence(self, binop):
        """ Propagate up the tree and return the first node with lower precedence than
            the given operator. """
        # If associativity is left-to-right, keep propagating up when precedence is equal.
        node = self
        while node.parent and node.parent.higher_precedence(binop):
            node = node.parent
        return node

    def leftmost_leaf(self):
        """ Propagate down the tree and returns the leftmost leaf of this node """
        if len(self.children) == 0 or self.associativity == Associativity.L_TO_R:
            # If the node is left-to-right associative, then it is considered to be
            # to the left of its child nodes
            return self

        return self.children[0].leftmost_leaf()

    def rightmost_leaf(self):
        """ Propagate down the tree and returns the rightmost leaf of this node """
        if len(self.children) == 0 or self.associativity == Associativity.R_TO_L:
            # If the node is right-to-left associative, then it is considered to be
            # to the right of its child nodes
            return self

        return self.children[-1].rightmost_leaf()

    def add_to_tree(self, tree, num):
        """ Add this syntax tree to a treelib.Tree. `num` is added to the tag
            to keep children in order. """
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
        for child in self.children:
            result = concat.func(result, child.evaluate(ctx))
        return result

    def __str__(self):
        return ', '.join(map(str, self.children))

    def __repr__(self):
        return '{}{}'.format(type(self).__name__, repr(self.children))

    def latex(self, ctx):
        return ',\, '.join(node.latex(ctx) for node in self.children)

    def _tree_tag(self):
        return '{}()'.format(type(self).__name__)


class EndOfExpression(Token):
    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        return node, i, None


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

        # Special case for concatenation operator. Instead of adding it
        # as a node, continue adding children to the parent node
        if binop.symbol == ',':
            return node.parent, i+1, binop.next_expected()

        node.insert_parent(binop)
        return binop, i+1, binop.next_expected()

    def next_expected(self):
        return [Parenthesis, UnaryOperator, Number, Declaration, Identifier]

    def evaluate(self, ctx):
        op = ctx.get(self.symbol, DefinitionType.BINARY_OPERATOR)
        return op.func(*self._eval_children(ctx, op))

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
        left = repr(self.children[0]) if len(self.children) > 0 else '?'
        right = repr(self.children[1]) if len(self.children) > 1 else '?'
        return 'BinaryOperator({}, {}, {})'.format(left, self.symbol, right)

    def latex(self, ctx):
        left = self.children[0]
        right = self.children[1]
        parens_left = self.is_left_parenthesized(left)
        parens_right = self.is_right_parenthesized(right)
        implicit = ImplicitMultiplication.is_implicit(self, left, right)

        definition = ctx.get(self.symbol, DefinitionType.BINARY_OPERATOR)

        if definition.latex_func:
            return definition.latex_func(
                definition, ctx, left, right,
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
        return op.func(*self._eval_children(ctx, op))

    def __str__(self):
        right = self.children[0]

        if self.is_right_parenthesized(right):
            right = '(' + str(right) + ')'

        return self.symbol + str(right)

    def __repr__(self):
        operand = repr(self.children[0]) if len(self.children) > 0 else '?'
        return 'UnaryOperator({}, {})'.format(self.symbol, operand)

    def latex(self, ctx):
        right = self.children[0]
        parens_right = self.is_right_parenthesized(right)

        definition = ctx.get(self.symbol, DefinitionType.UNARY_OPERATOR)

        if definition.latex_func:
            return definition.latex_func(self, ctx, right, parens_right)

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
            # If the current node is a function, keep the working node on the function.
            # Add all child nodes of the parenthesized expression root as children of the
            # function.
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
        if isinstance(root, (Identifier, Declaration)) and not root.is_const:
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
        if node.n_args == 0:
            # If the function is 0-arg, remove any empty parentheses
            if expr[i:i+2] == '()':
                i += 2
            return node, i, cls.next_expected(explicit=True)

        if expr[i] == '(':
            # Parse the inside of the parentheses
            node, i, _ = Parenthesis.parse(ctx, node, i, expr, start, end)
            return node, i, cls.next_expected(explicit=True)

        # Implicit function call
        return node, i, cls.next_expected()

    @classmethod
    def next_expected(cls, explicit=False):
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
        decimal = False
        j = i
        while j < end:
            ch = expr[j]
            if ch.isdigit():
                # Eat a digit
                n += ch
            elif not decimal and ch == '.':
                # Eat a decimal point
                # ensure that only one decimal point exists
                n += ch
                decimal = True
            else:
                break
            j += 1

        # Invalid number.
        if n == '' or n == '.':
            return node, i, None

        # Convert string to float/int
        n = float(n)
        if n % 1 == 0:
            n = int(n)

        num = cls(n)
        node.add_child(num)
        return num, j, num.next_expected()

    def next_expected(self):
        return [BinaryOperator, ImplicitMultiplication, EndOfExpression]

    def evaluate(self, ctx:Context):
        return self.value

    def __repr__(self):
        return 'Number({})'.format(self.value)

    def __str__(self):
        return str(self.value)

    def latex(self, ctx):
        return str(self.value)

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.value)


class Identifier(Node):
    def __init__(self, definition:Definition):
        super().__init__(definition.precedence, definition.associativity)
        self.name = definition.name
        self._display_name = getattr(definition, 'display_name', None)
        self.n_args = len(definition.args)
        self.is_const = definition.is_constant

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        j = i
        string = ''
        definition = None

        # Find the longest continuous identifier that exists in the current context
        # Due to implicit multiplication & such, identifiers may be strung together,
        # ex. "sincosepi" = "sin(cos(e*pi))"
        # `string` will contain the entire string "sincosepi", and j will be the end
        # index of `string`. Definition will be the definition of the first identifier
        # in the context, in this example the definition for "sin".
        # Parsing the rest ("cosepi") will be delegated to the next token.
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
        args = [repr(self.children[i]) if i < len(self.children) else '?'
                for i in range(self.n_args)]
        return '{}({}, {})'.format(type(self).__name__, self.name, ', '.join(args))

    def latex(self, ctx):
        raise NotImplemented

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.name, self.n_args)


class Function(Identifier):
    def next_expected(self):
        return [BinaryOperator, FunctionCall, EndOfExpression]

    def evaluate(self, ctx:Context):
        definition:Definition = ctx.get(self.name, DefinitionType.IDENTIFIER)

        if len(self.children) == 0 and len(definition.args) > 0:
            # If the function takes arguments but was given none,
            # return the definition itself.
            return definition

        # Evaluate arguments & pass it to the function
        inputs = list(self._eval_children(ctx, definition))
        definition.check_inputs(len(inputs))
        return definition(*inputs)

    def __str__(self):
        name = self._display_name or self.name
        if self.n_args > 0 and len(self.children) == 0:
            return name

        def arg_str(child):
            if isinstance(child, ListNode):
                return '(' + str(child) + ')'
            return str(child)

        args = ', '.join(map(arg_str, self.children))
        return '{}({})'.format(name, args)

    def latex(self, ctx):
        definition = ctx.get(self.name)
        n_inputs = len(self.children)

        if n_inputs == 0 and len(definition.args) > 0:
            # No function call
            return replace_latex_symbols(definition.signature)

        if definition.manual_eval:
            # add phantom `ctx` input for manual_eval functions
            n_inputs += 1
        definition.check_inputs(n_inputs)

        if definition.latex_func:
            return definition.latex_func(self, ctx, *self.children)

        return r'{}\left( {} \right)'.format(
            replace_latex_symbols(self.name),
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
        return '{}({})'.format(type(self).__name__, self.name)

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.name)

    def latex(self, ctx):
        definition = ctx.get(self.name)
        return getattr(definition, 'latex_name', None) \
            or replace_latex_symbols(definition.display_name or definition.name)


class ImplicitMultiplication(Token):
    # after: number, variable
    # before: parenthesis, number, variable
    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        # if isinstance(node, (Function, Declaration)) and len(node.children) == 0:
        #     # Functions followed by implicit multiplication is a call if
        #     # the function has no children.
        #     # Variables followed by '()' are also not implicit multiplication;
        #     # that will be handled by Parenthesis later.
        #     return node, i, None

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

        # if isinstance(left.rightmost_leaf(), Identifier) \
        #         and isinstance(right.leftmost_leaf(), Identifier):
        #     # Don't implicitly multiply two identifiers
        #     return False

        return True


class Declaration(Node):
    """ A declaration of a new identifier.
        Takes the form of foo(a,b,c)=... """
    precedence = 1

    def __init__(self, definition, root):
        super().__init__()
        self.definition = definition
        self.n_args = len(definition.args)
        self.is_const = definition.is_constant
        self.root = root

    @classmethod
    def parse(cls, ctx, node, i, expr, start, end):
        # Index of equals sign
        equals = expr.find('=', i, end)

        if equals == -1:
            # Not a declaration.
            return node, i, None

        try:
            # Parse the signature
            name, args, is_const = cls.parse_signature(expr, i, equals)
        except StopIteration:
            # If signature parsing fails, this isn't a valid declaration.
            # Probably means there's a declaration later on and this is an identifier.
            return node, i, None

        # Create a function definition
        definition = DeclaredFunction(name, args, is_const)

        # Push the function definition and its argument variables to the context
        # and parse the remainder of the expression
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
        Raises StopIteration if the signature syntax is invalid.

        Returns (name, args, const).
        `args` is a list of strings, see FunctionDefinition().
        `is_const` is whether this declaration should be evaluated once then saved, or re-evalated
        each time it is used. This is True if the signature takes no arguments and does not have
        empty parentheses. For example "x() = 3y" would change when y changes, whereas "x = 3y"
        would not.)

        :param expr: Expression string
        :param i: Current index
        :param end: End index of the signature
        :return: identifier name, arg list, bool const
        """
        # index of opening parenthesis
        parens = expr.find('(', i, end)

        if parens == -1:
            # Variable declaration (0-arg function) without parentheses.
            # Entire signature must be a single identifier.
            parens = end
            rparens = end - 1
            is_const = True
        else:
            # Locate closing parenthesis. Should be before the equals sign.
            # If it isn't, this may be a different token followed by a declaration
            # somewhere later.
            rparens = Parenthesis.find_close(expr, parens, end, err=False)
            is_const = False

        if rparens == -1 or rparens != end - 1:
            # Closing parenthesis exists but it is not immediately before the equals sign.
            # This token probably isn't a declaration. Ex:
            raise StopIteration

        # Name of declared identifier
        name = expr[i:parens]
        if not is_identifier(name):
            raise StopIteration

        # Move start & end to capture the argument list
        i = parens + 1
        end = rparens

        # Parse argument list
        args = []
        if i < end:
            # More than 0 arguments
            j = i
            while j < end:
                if expr[j] == ',':
                    args.append(expr[i:j])
                    i = j + 1
                j+=1
            # last arg
            args.append(expr[i:j])

        return name, args, is_const

    @classmethod
    def find_expression_end(cls, ctx, expr, start, end):
        """
        Given a declaration, determines the end of the expression after the equals sign.
        For example, in the expression "f(x)=3(3+2), 5, 7" the declaration for `f` could be
        "3+2" or "3+2, 5, 7" depending on precedence.
        Current implementation: Stop at first binary operator with greater precedence than
        Declaration

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
                i = Parenthesis.find_close(expr, i, end)
            elif (ch, DefinitionType.BINARY_OPERATOR) in ctx:
                binop = ctx.get(ch, DefinitionType.BINARY_OPERATOR)
                if cls.precedence > binop.precedence:
                    break
            i += 1
        return i

    def next_expected(self):
        return [BinaryOperator, FunctionCall, EndOfExpression]

    def evaluate(self, ctx:Context):
        self.definition.bind_context(ctx)
        if len(self.children) > 0:
            inputs = self._eval_children(ctx, self.definition)
            return self.definition(*inputs)
        else:
            return self.definition

    def __str__(self):
        if len(self.children) == 0:
            return str(self.definition)

        args = ', '.join(map(str, self.children))
        return '({})({})'.format(self.definition, args)

    def __repr__(self):
        return '{}({}, [{}], {})'.format(
            type(self).__name__,
            self.definition.name,
            ', '.join(self.definition.args),
            repr(self.root)
        )

    def latex(self, ctx):
        with ctx.with_scope():
            if not self.definition.name in ctx:
                ctx.add(self.definition)
            self.definition.add_args_to_context(ctx, None)

            signature = self.definition.signature
            signature = replace_latex_symbols(signature)
            body = self.definition.func.latex(ctx)
            if isinstance(self.definition.func, ListNode):
                body = r'\left( ' + body + r' \right)'

            result = '{} = {}'.format(signature, body)

            if len(self.children) > 0:
                # Lambda call "(f(x)=3x)(6)"
                result = r'\left( {} \right)\left( {} \right)'.format(
                    result,
                    ',\, '.join(node.latex(ctx) for node in self.children)
                )

        return result

    def _tree_tag(self):
        return '{}({})'.format(type(self).__name__, self.definition.signature)

    def add_to_tree(self, tree, num):
        print('Declaration', str(self.definition))
        tree2 = type(tree)() # make a new tree
        self.root.add_to_tree(tree2, 0)
        tree2.show()
        super().add_to_tree(tree, num)
