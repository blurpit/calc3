import pickle
from contextlib import contextmanager

from .definitions import DefinitionType, Definition, DeclaredFunction


class ContextError(Exception):
    pass

class Params:
    # Number of decimal places to round numbers to after evaluating. Applies
    # to floats, vectors, & matrices.
    #  - None (default behavior) means no rounding, leave results as is. This
    #    may result in lots of floating point precision errors.
    #  - If rounding is enabled, floats with no decimal component will be converted
    #    into ints.
    rounding = None

    # If False, identifiers that don't exist in the context at parse time
    # will raise an ExpressionSyntaxError (default behavior). If True, unknown
    # identifiers will be added to the syntax tree as Variables.
    #  - Note that the parser will not try to guess how best to split unknown
    #    identifiers and will simply read left to right; for example if 'x' is
    #    defined, 'xy' will be parsed as 'x*y' but 'yx' will be parsed as a
    #    single variable 'yx'.
    parse_unknown_identifiers = False

class Context:
    def __init__(self):
        self.params = Params()
        self.stack = [{}]
        self.ans = 0

    def add(self, *items:Definition, override_global=False):
        """ Add a collection of Definitions to the top scope in the context. If `override_global`
            is False, modifying the global scope or overriding items from the global scope will
            raise a ContextError. """
        for item in items:
            self.set(item.name, item.token_type, item, override_global)

    def get(self, name:str, token_type:DefinitionType=DefinitionType.IDENTIFIER, default=ContextError):
        """ Get an item from the context. Items higher on the scope stack will be returned first. """
        for ctx in reversed(self.stack):
            result = ctx.get((name, token_type), ContextError)
            if result is not ContextError:
                return result

        if default is ContextError:
            raise ContextError("'{}' is undefined.".format(name))
        else:
            return default

    def __contains__(self, item):
        if isinstance(item, tuple):
            if len(item) != 2:
                return False
        elif isinstance(item, Definition):
            item = (item.name, item.token_type)
        else:
            item = (item, DefinitionType.IDENTIFIER)

        for scope in reversed(self.stack):
            if item in scope:
                return True
        return False

    def set(self, name:str, token_type:DefinitionType, val:Definition, override_global=False):
        """ Set an item in the context directly. In most cases you should use ``add()`` instead. """
        if not override_global:
            if len(self.stack) < 2:
                raise ContextError('Cannot modify global scope')
            elif (name, token_type) in self.stack[0]:
                raise ContextError("Cannot override '{}' from global scope".format(name))

        if isinstance(val, DeclaredFunction):
            # If adding a DeclaredFunction, make a duplicate of the definition and
            # bind it to this context
            if val.ctx is not None and val.ctx is not self:
                val = val.copy()
            val.bind_context(self)

        self.stack[-1][(name, token_type)] = val

    def keys(self):
        result = set()
        for scope in self.stack:
            result.update(scope.keys())
        return result

    def push_scope(self):
        """ Push a new scope to the stack """
        self.stack.append({})

    def pop_scope(self):
        """ Pop the top scope off the stack """
        if len(self.stack) > 1:
            del self.stack[-1]
        else:
            raise ContextError('Cannot pop global scope')

    @contextmanager
    def with_scope(self):
        """ Push a scope using a context manager """
        try:
            self.push_scope()
            yield
        finally:
            self.pop_scope()

    def save(self, filename):
        """ Save DeclaredFunctions in the context to a file """
        ctx = []
        for scope in self.stack[1:]:
            items = []
            for definition in scope.values():
                if isinstance(definition, DeclaredFunction):
                    # Unbind the context because it cannot be pickled
                    definition.bind_context(None)
                    items.append(definition)
            if items:
                ctx.append(items)

        # Write to the file
        with open(filename, 'wb') as file:
            pickle.dump(ctx, file)

        # Re-bind declared functions context
        for items in ctx:
            for definition in items:
                definition.bind_context(self)

    def load(self, filename):
        """ Load a context from a file created using ``ctx.save()`` """
        with open(filename, 'rb') as file:
            ctx = pickle.load(file)

        for scope in ctx:
            self.push_scope()
            for definition in scope:
                self.add(definition)

    def round_result(self, result):
        """ Rounds a result according to params.rounding """
        if self.params.rounding is not None:
            if hasattr(result, '__round__'):
                # Object has its own round function
                result = round(result, self.params.rounding)
                if isinstance(result, float) and result % 1 == 0:
                    result = int(result)
            elif type(result) == list:
                # Round each element of the list
                for i, x in enumerate(result):
                    result[i] = self.round_result(x)
        return result

    def __len__(self):
        """ Number of scopes in this context """
        return len(self.stack)

    def __str__(self):
        s = 'Context(\n'
        for i, ctx in enumerate(self.stack[1:], 1):
            for item in ctx.values():
                s += '\t'*i + str(item) + '\n'
        s += ')'
        return s
