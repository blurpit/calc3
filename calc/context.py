from contextlib import contextmanager

from .definitions import DefinitionType, Definition, DeclaredFunction


class ContextError(Exception):
    pass

class Context:
    def __init__(self):
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

    def __len__(self):
        return len(self.stack)

    def __str__(self):
        s = 'ctx('
        for i, ctx in enumerate(self.stack):
            s += '\n\t' + str(i) + ': ' + ' '.join(ctx.keys())
        return s + '\n)'
