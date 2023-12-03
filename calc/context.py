import math
import pickle
import sys
from contextlib import contextmanager
from typing import List, Tuple, Union

from .definitions import DeclaredFunction, Definition, DefinitionType


class ContextError(Exception):
    pass


class Params:
    # Number of decimal places to round numbers to after evaluating. Applies
    # to floats, vectors, & matrices.
    #  - None (default behavior) means no rounding, leave results as is. This
    #    may result in floating point precision errors.
    #  - Floats with no decimal component will still be converted to ints even
    #    if rounding is disabled.
    rounding = None

    # If False, identifiers that don't exist in the context at parse time
    # will raise an ExpressionSyntaxError (default behavior). If True, unknown
    # identifiers will be added to the syntax tree as Variables.
    #  - Note that the parser will not try to guess how best to split unknown
    #    identifiers and will simply read left to right; for example if 'x' is
    #    defined, 'xy' will be parsed as 'x*y' but 'yx' will be parsed as a
    #    single variable 'yx'.
    parse_unknown_identifiers = False

    # If False, any items added to the context with the same name (and definition
    # type) as an item in the global scope will raise a ContextError. Shadowed items
    # replace an item in its parent scope(s) until its scope is popped.
    allow_global_scope_shadowing = False

    # If True, the scope
    save_declared_function_scopes = True


class Scope(dict):
    def add(self, definition: Definition):
        """ Add a definition to the scope. Replaces the existing definition if there is one. """
        self[(definition.name, definition.token_type)] = definition

    def get(self, name: str, token_type: DefinitionType = DefinitionType.IDENTIFIER, default=None):
        """ Get a definition from the scope. """
        return super().get((name, token_type), default)

    def __contains__(self, item: Union[Definition, Tuple[str, DefinitionType]]):
        if isinstance(item, Definition):
            return super().__contains__((item.name, item.token_type))
        elif isinstance(item, tuple):
            return super().__contains__(item)
        else:
            return False

    def __str__(self):
        s = 'Scope {\n'
        for definition in self.values():
            s += '\t' + str(definition) + '\n'
        s += '}'
        return s

    def __repr__(self):
        return '<Scope size={}>'.format(len(self))


class Context:
    def __init__(self):
        self.params = Params()
        self.global_scope = Scope()
        self.stack: List[Scope] = [self.global_scope]
        self.ans = 0

    def add(self, *definitions: Definition):
        """ Add a collection of Definitions to the top scope in the context. """
        # If the global scope is the only scope, raise an error. Use ctx.add_global to modify the
        # global scope instead.
        if len(self.stack) == 1:
            raise ContextError('Cannot modify global scope')

        for definition in definitions:
            # Check if the definition would shadow the global scope
            if not self.params.allow_global_scope_shadowing and definition in self.global_scope:
                raise ContextError("Cannot shadow '{}' from global scope".format(definition.name))

            # If the definition is bound to another context, make a duplicate and bind it to this context
            if definition.ctx is not None and definition.ctx is not self:
                definition = definition.copy()
            definition.bind_context(self)

            # Add the definition to the scope at the top of the stack
            self.stack[-1].add(definition)

    def add_global(self, *definitions: Definition):
        """ Add a collection of definitions to the global scope. """
        for definition in definitions:
            # If the definition is bound to another context, make a duplicate and bind it to this context
            if definition.ctx is not None and definition.ctx is not self:
                definition = definition.copy()
            definition.bind_context(self)

            self.global_scope.add(definition)

    def get(self, name: str, token_type: DefinitionType = DefinitionType.IDENTIFIER, default=ContextError):
        """ Get an item from the context. Items higher on the scope stack will be returned first. """
        for scope in reversed(self.stack):
            result = scope.get(name, token_type)
            if result is not None:
                return result

        if default is ContextError:
            raise ContextError("'{}' is undefined.".format(name))
        else:
            return default

    def __contains__(self, item):
        for scope in reversed(self.stack):
            if item in scope:
                return True
        return False

    def keys(self):
        result = set()
        for scope in self.stack:
            result.update(scope.keys())
        return result

    def remove(self, name: str, token_type: DefinitionType):
        """ Removes and returns an item from the context given the name. If the name appears more than
            once in the context, removes the first occurance moving down from the top of the stack.
            Raises a ContextError if the name is not found or is in the global scope. """
        key = (name, token_type)
        for scope in reversed(self.stack[1:]):
            result = scope.pop(key, None)
            if result is not None:
                return result

        if key in self.stack[0]:
            # name is in the global scope
            raise ContextError("Cannot override '{}' from global scope".format(name))
        else:
            # name is not in the context
            raise ContextError("'{}' is undefined.".format(name))

    def condense(self):
        """ Returns a Scope containing all non-global items in this context (removes shadowed items). """
        condensed = Scope()
        for scope in self.stack[1:]:
            for k, v in scope.items():
                condensed[k] = v
        return condensed

    def push_scope(self):
        """ Push a new scope to the stack """
        self.stack.append(Scope())

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

    @contextmanager
    def with_inserted_scope(self, scope=None, stack_index=-1):
        """ Push a specific given scope *after* a particular index in the stack """
        if scope is None:
            scope = Scope()
        if stack_index < 0:
            stack_index += len(self.stack)
        stack_index += 1

        try:
            self.stack.insert(stack_index, scope)
            yield
        finally:
            self.stack.pop(stack_index)

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
        """ Rounds a result according to ``params.rounding`` """
        if self.params.rounding is not None:
            if hasattr(result, '__round__'):
                # Object has its own round function
                result = round(result, self.params.rounding)
            elif isinstance(result, list):
                # Round each element of the list
                result = result.copy()
                for i, x in enumerate(result):
                    result[i] = self.round_result(x)

        # If the result is too precise, it means there is some ambiguity whether the "true" value has been represented
        # accuratly. If there's a possibility some information was lost, we should keep it as a float to signify that
        # possibility, and to avoid strangeness that occurs when converting between ints and imprecise floats.
        # For example, int(1e24) equals 999999999999999983222784 in python.
        if isinstance(result, float) and result % 1 == 0 and not self._too_precise(result):
            result = int(result)

        return result

    @staticmethod
    def _too_precise(n: float):
        """ Returns true if the whole number part of a float `n` is too precise to fit in its mantissa """
        m, e = math.frexp(n)
        dig = sys.float_info.mant_dig
        return abs(e) > dig

    def __len__(self):
        """ Number of scopes in this context """
        return len(self.stack)

    def __str__(self):
        s = 'Context {\n'
        s += '\t<global scope: {} items>\n'.format(len(self.global_scope))
        for i, scope in enumerate(self.stack[1:], 1):
            for definition in scope.values():
                s += '\t' * i + str(definition) + '\n'
        s += '}'
        return s

    def __repr__(self):
        return '<Context size={}>'.format(len(self.stack))
