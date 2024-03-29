"""Property decorators."""


class cached_property(object):
    """Decorator for lazily loading attributes.

    With this, the call to a class function becomes an attribute
    myobject.function and NOT myobject.function()
    The attribute is calculated when the function is called and stored
    as a property for later speedups.
    """

    def __init__(self, func):
        self._attr_name = func.__name__  # grabs the name of the decorated func
        self._func = func

    def __get__(self, instance, owner):
        attr = self._func(instance)
        setattr(instance, self._attr_name, attr)
        return attr
