"""
6.101 Lab:
LISP Interpreter Part 2
"""

#!/usr/bin/env python3
import sys
import operator
import functools
import re


sys.setrecursionlimit(20_000)

# KEEP THE ABOVE LINES INTACT, BUT REPLACE THIS COMMENT WITH YOUR lab.py FROM
# THE PREVIOUS LAB, WHICH SHOULD BE THE STARTING POINT FOR THIS LAB.

# import operator
# import math

sys.setrecursionlimit(20_000)

# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """

    if scheme_booleanp(value):
        return value
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """

    tokens = []
    current_position = 0

    def tokenize_comment():
        nonlocal current_position
        while current_position < len(source):
            if source[current_position] == "\n":
                current_position += 1
                break
            current_position += 1

    def tokenize_symbol():
        nonlocal current_position
        token = []
        while current_position < len(source):
            if source[current_position] in " ()\n":
                # current_position += 1
                return "".join(token)
            token.append(source[current_position])
            current_position += 1
        return "".join(token)

    while current_position < len(source):
        ch = source[current_position]
        if ch.isspace():
            current_position += 1
        elif ch == ";":
            tokenize_comment()
        elif ch in "()":
            tokens.append(ch)
            current_position += 1
        elif ch.isascii():
            tokens.append(tokenize_symbol())
        else:
            raise ValueError(f"Invalid charater '{ch}' at position {current_position}")
    trans = {"#t": True, "#f": False}
    return list(map(lambda x: trans[x] if x in trans else x, tokens))


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    """
        expr ::= atom | s-expr 
        s-expr ::= "(" expr ")"
    """

    def parse_expr(tokens):
        token = tokens[0]
        if scheme_booleanp(token) or token not in "()":
            return number_or_symbol(token), tokens[1:]
        elif token == "(":
            s_expr = []
            tokens = tokens[1:]
            while True:
                if len(tokens) == 0:
                    raise SchemeSyntaxError
                if tokens[0] == ")":
                    return s_expr, tokens[1:]
                token, tokens = parse_expr(tokens)
                s_expr.append(token)
        else:
            raise SchemeSyntaxError

    if len(tokens) != 0:
        tree, rest = parse_expr(tokens)
        if len(rest) != 0:
            raise SchemeSyntaxError
        return tree if tree is not None else None


######################
#        util        #
######################
_PREFIX = ""


def log(message):
    """Print an indented message (used with trace)."""
    print(_PREFIX + re.sub("\n", "\n" + _PREFIX, str(message)))


def trace(fn):
    """A decorator that prints a function's name, its arguments, and its return
    values each time the function is called. For example,

    @trace
    def compute_something(x, y):
        # function body
    """

    @functools.wraps(fn)
    def wrapped(*args, **kwds):
        global _PREFIX
        reprs = [repr(e) for e in args]
        reprs += [repr(k) + "=" + repr(v) for k, v in kwds.items()]
        log("{0}({1})".format(fn.__name__, ", ".join(reprs)) + ":")
        _PREFIX += "     "
        try:
            result = fn(*args, **kwds)
            _PREFIX = _PREFIX[:4]
        except Exception as e:
            log(fn.__name__ + " exited via exception")
            _PREFIX = _PREFIX[:4]
            raise
        log("{0}({1}) -> {2}".format(fn.__name__, ", ".join(reprs), result))
        return result

    return wrapped


def scheme_symbolp(x):
    return isinstance(x, str)


def scheme_booleanp(x):
    return x is True or x is False


def scheme_numberp(x):
    return isinstance(x, int | float) and not scheme_booleanp(x)


def scheme_nullp(x):
    return type(x).__name__ == "nil"


def is_scheme_true(val):
    return val is not False


def is_scheme_false(val):
    return val is False


def scheme_atomp(x):
    return (
        scheme_booleanp(x) or scheme_numberp(x) or scheme_symbolp(x) or scheme_nullp(x)
    )


def self_evaluating(expr):
    return scheme_atomp(expr) and not scheme_symbolp(expr)


def scheme_listp(x):
    return isinstance(x, list)


def scheme_pairp(x):
    return type(x).__name__ == "Pair"


def scheme_procedurep(x):
    return isinstance(x, Procedure)


def validate_procedure(procedure):
    """
    Check that PORCEDURE is a valid Scheme porcedure.
    """
    if not scheme_procedurep(procedure):
        raise SchemeEvaluationError(
            "{0} is not callable: {1}".format(
                type(procedure).__name__.lower(), repr(procedure)
            )
        )


def validate_type(val, predicate, k, name):
    """
    Return VAL. Raise a SchemeEvaluateError if not PREDICATE(VAl)
    using
    """
    if not predicate(val):
        msg = "argument {0} of {1} has wrong type ({2})"
        type_name = type(val).__name___
        if scheme_symbolp(val):
            type_name = "symbol"
        raise SchemeEvaluationError(msg.format(k, name, type_name))
    return val


def validate_form(expr, min, max=float("inf")):
    """
    Check EXPR is a proper list whose length is at least MIN and no more than MAX (default: no maximum).
    Raise a SchemeError if this is not the case.
    """
    if not scheme_listp(expr):
        raise SchemeError(f"badly formed expression: {expr}")
    length = len(expr)
    if length < min:
        raise SchemeError("too few operands in form")
    elif length > max:
        raise SchemeError("too many operands in form")


def validate_formals(formals):
    """
    Check that FORMALS is a valid parameter list, a list of symbols in which
    each symbol is distinct. Raise a SchemeErro if the list of formals is not
    a a list of symbols or if any symbols is repeated.
    """
    symbols = set()

    def validate_and_add(symbol, is_last):
        if not scheme_symbolp(symbol):
            raise SchemeError("non-symbol: {0}".format(symbol))
        if symbol in symbols:
            raise SchemeError("duplicate symbol: {0}".format(symbol))

        symbols.add(symbol)

    while isinstance(formals, list):
        validate_and_add(formals[0], len(formals[1:]) == 0)
        formals = formals[1:]


######################
# Built-in Functions #
######################

BUILTINS = []


def builtin(*names, need_env=False):
    """
    An annotation to convert a Python function into a BilltProcedure.
    """

    def add(py_func):
        for name in names:
            BUILTINS.append((name, py_func, names[0], need_env))
        return py_func

    return add


def _check_nums(*vals):
    """
    Check that all arguments in Vals are numbers.
    """
    for i, v in enumerate(vals):
        if not scheme_numberp(v):
            msg = f"operand {i} ({v}) is not a number"
            raise SchemeError(msg)


def _arith(fn, init, vals):
    """
    Perform the FN operator on the numebr of values of VALS, with init as the value
    when VALs is empty, Return the result as a Scheme value.
    """
    _check_nums(*vals)
    s = init
    for val in vals:
        s = fn(s, val)

    return s


## Computing operator
@builtin("+")
def scheme_add(*vals):
    return _arith(operator.add, 0, vals)


@builtin("-")
def scheme_sub(val0, *vals):
    if len(vals) == 0:
        return -val0
    return _arith(operator.sub, val0, vals)


@builtin("*")
def scheme_mul(*vals):
    return _arith(operator.mul, 1, vals)


@builtin("/")
def scheme_div(val0, *vals):
    try:
        if len(vals) == 0:
            return operator.truediv(1, val0)
        return _arith(operator.truediv, val0, vals)

    except ZeroDivisionError as err:
        raise SchemeError(err)


## Comparise
@builtin("not")
def scheme_not(*vals):
    if len(vals) != 1:
        raise SchemeEvaluationError
    return not is_scheme_true(vals[0])


@builtin("equal?")
def scheme_equalp(val0, *vals):
    return all(map(lambda x: val0 == x, vals))


@builtin("eq?")
def scheme_eqp(x, y):
    if scheme_numberp(x) and scheme_numberp(y):
        return x == y
    elif scheme_symbolp(x) and scheme_symbolp(y):
        return x == y
    else:
        return x is y


def _numcomp(op, x, y):
    _check_nums(x, y)
    return op(x, y)


@builtin("=")
def scheme_eq(x, y):
    return _numcomp(operator.eq, x, y)


@builtin("<")
def scheme_lt(val0, *vals):
    if len(vals) == 0:
        return True
    return all(map(lambda x: _numcomp(operator.lt, val0, x), vals)) and scheme_lt(
        vals[0], *vals[1:]
    )


@builtin(">")
def scheme_gt(val0, *vals):
    if len(vals) == 0:
        return True
    return all(map(lambda x: _numcomp(operator.gt, val0, x), vals)) and scheme_gt(
        vals[0], *vals[1:]
    )


@builtin("<=")
def scheme_le(val0, *vals):
    if len(vals) == 0:
        return True
    return all(map(lambda x: _numcomp(operator.le, val0, x), vals)) and scheme_le(
        vals[0], *vals[1:]
    )


@builtin(">=")
def scheme_ge(val0, *vals):
    if len(vals) == 0:
        return True
    return all(map(lambda x: _numcomp(operator.ge, val0, x), vals)) and scheme_ge(
        vals[0], *vals[1:]
    )


@builtin("cons")
def scheme_cons(*vals):
    if len(vals) != 2:
        raise SchemeEvaluationError
    return Pair(vals[0], vals[1])


@builtin("list")
def scheme_list(*vals):
    result = nil
    for e in reversed(vals):
        result = Pair(e, result)
    return result


@builtin("car")
def scheme_car(*x):
    # validate_type(x, scheme_pairp, 0, "car")
    if len(x) != 1 or (not isinstance(x[0], Pair)):
        raise SchemeEvaluationError
    return x[0].car


@builtin("cdr")
def scheme_cdr(*x):
    # validate_type(x, scheme_pairp, 0, "cdr")
    if len(x) != 1 or (not isinstance(x[0], Pair)):
        raise SchemeEvaluationError
    return x[0].cdr


@builtin("list?")
def listp(*x):
    if len(x) != 1:
        raise SchemeEvaluationError
    x = x[0]
    while x is not nil:
        if not isinstance(x, Pair):
            return False
        x = x.cdr
    return True


@builtin("length")
def scheme_length(*x):
    if len(x) != 1:
        raise SchemeEvaluationError
    x = x[0]
    if not listp(x):
        raise SchemeEvaluationError
    return len(x)


@builtin("list-ref")
def scheme_list_ref(*x):
    if len(x) != 2:
        raise SchemeEvaluationError
    ll, index = x[0], x[1]
    if not scheme_pairp(ll) or not isinstance(index, int):
        raise SchemeEvaluationError
    if listp(ll):
        if len(ll) <= index:
            raise SchemeEvaluationError
        while index != 0:
            index = index - 1
            ll = ll.cdr
        return ll.car
    elif scheme_pairp(ll):
        if index != 0:
            raise SchemeEvaluationError
        return ll.car
    else:
        raise SchemeEvaluationError


@builtin("append")
def scheme_append(*vals):
    if len(vals) == 0:
        return nil
    result = vals[-1]
    if not listp(result):
        raise SchemeEvaluationError
    for i in range(len(vals) - 2, -1, -1):
        v = vals[i]
        if v is not nil:
            # validate_type(v, scheme_pairp, i, "append")
            if not listp(v):
                raise SchemeEvaluationError
            r = p = Pair(v.car, result)
            v = v.cdr
            while scheme_pairp(v):
                p.cdr = Pair(v.car, result)
                p = p.cdr
                v = v.cdr
            result = r
    return result


@builtin("begin", need_env=True)
def scheme_begin(*exprs, env):
    if len(exprs) == 0:
        raise SchemeEvaluationError
    for expr in exprs[:-1]:
        evaluate(expr, env)
    result = evaluate(exprs[-1], env)
    return result


builtin("null?")(scheme_nullp)


def add_builtins(frame, funcs_and_names):
    for name, py_func, proc_name, need_env in funcs_and_names:
        frame.define(name, BuiltinProcedure(py_func, name=proc_name, need_env=need_env))


def create_global_frame():
    """
    Initialize and return a single-frame enviornment with built-in names.
    """
    env = Frame(None)
    add_builtins(env, BUILTINS)
    return env


######################
#    special_forms   #
######################


def do_define_form(expression, env):
    validate_form(expression, 2)
    signature = expression[0]
    if scheme_symbolp(signature):
        # assign a name to a value e.g (define x (+ 1 2))
        validate_form(expression, 2, 2)
        value = evaluate(expression[1], env)
        env.define(signature, value)
        return value
    elif scheme_listp(signature) and scheme_symbolp(signature[0]):
        # define a named procedure e.g (define (f x y) (+ x y)) or (define f (lambda (x y) (+ x y)))
        validate_form(expression, 2, 2)
        name = signature[0]
        procedure = LambdaProcedure(signature[1:], expression[1], env)
        env.define(name, procedure)
        return "function object"
    else:
        bad_signature = signature[0] if isinstance(signature, list) else signature
        raise SchemeError("non-symbol: {0}".format(bad_signature))


def do_lambda_form(expression, env):
    """
    Evaluate a lambda form
    """
    # define a lambda procedure e.g (lambda (x y) (+ x y))
    validate_form(expression, 2, 2)
    return LambdaProcedure(expression[0], expression[1], env)


def do_if_form(expression, env):
    """
    Evaluate an if form
    """
    # e.g (if (equal? 1 n) n m)
    validate_form(expression, 2, 3)
    if is_scheme_true(evaluate(expression[0], env)):
        return evaluate(expression[1], env)
    elif len(expression) == 3:
        return evaluate(expression[2], env)


def do_and_form(expression, env):
    """
    Evaluate a (short-circuited) and form
    """
    # e.g (and (> 2 3) #f #f)
    validate_form(expression, 1)
    for expr in expression:
        result = evaluate(expr, env)
        if is_scheme_false(result):
            return False
    return True


def do_or_form(expression, env):
    """
    Evaluate a (short-circuited) or form
    """
    # e.g (or (> 2 3) #f #t)
    validate_form(expression, 1)
    for expr in expression:
        result = evaluate(expr, env)
        if is_scheme_true(result):
            return True
    return False


def do_del_form(expression, env):
    """
    Evaluate a del form
    """
    # e.g (del x)
    validate_form(expression, 1, 1)
    value = env.delete(expression[0])
    return value


def do_let_form(expression, env):
    """
    Evaluate a let form
    """
    # e.g (let ((val1 1) (val2 2) ...) (body))
    validate_form(expression, 2, 2)
    bindings = expression[0]
    body = expression[1]
    let_frame = Frame(env)
    for bind in bindings:
        validate_form(bind, 2, 2)
        let_frame.define(bind[0], evaluate(bind[1], env))
    return evaluate(body, let_frame)


def do_set_form(expression, env):
    """
    Evaluate a set form
    """
    # e.g (set! var expression)
    validate_form(expression, 2, 2)
    symbol = expression[0]
    in_which = env.look_frame(symbol)
    val = evaluate(expression[1], env)
    in_which.define(symbol, val)
    return val


scheme_special_forms = {
    "define": do_define_form,
    "lambda": do_lambda_form,
    "if": do_if_form,
    "and": do_and_form,
    "or": do_or_form,
    "del": do_del_form,
    "let": do_let_form,
    "set!": do_set_form,
}


##############
#    Frame   #
##############


class Frame:
    """
    An environment frame binds Scheme symbols to Scheme values.
    """

    def __init__(self, parent):
        self.bindings = {}
        self.parent = parent

    def __repr__(self):
        if self.parent is None:
            return "<Global Frame>"
        s = sorted(["{0}: {1}".format(k, v) for k, v in self.bindings.items()])
        return "<{{{0}}} -> {1}>".format(", ".join(s), repr(self.parent))

    def __str__(self):
        return "<Frame>"

    def define(self, symbol, value):
        # if symbol in self.bindings:
        #     raise SchemeNameError(f"duplicate symbol: {symbol}")
        self.bindings[symbol] = value

    def look_frame(self, symbol):
        """
        Finding the nearest enclosing frame in which symbol is defined.
        """
        current = self
        while current is not None:
            value = current.bindings.get(symbol, None)
            if value is not None:
                return current
            current = current.parent
        raise SchemeNameError(f"unknown identifier: {symbol}")

    def lookup(self, symbol):
        """
        Return the value bound of SYMBOL, Errors if SYMBOL is not found.
        """
        env = self.look_frame(symbol)
        return env.bindings[symbol]

    def delete(self, symbol):
        """
        Deleting variable bindings with the current frame, raise SchemeNameError if the variable don't exist
        in the current frame, and return the associated value.
        """
        value = self.bindings.get(symbol, None)

        if value is not None:
            self.bindings.pop(symbol)
            return value
        raise SchemeNameError


def make_initial_frame():
    return Frame(global_frame)


##############
# Procedures #
##############


class Procedure:
    """
    The base class for all Procedure classes.
    """

    pass


class BuiltinProcedure(Procedure):
    """
    A Scheme procedure defined as a python function.
    """

    def __init__(self, py_func, need_env=False, name="builtin"):
        self.name = name
        self.py_func = py_func
        self.need_env = need_env

    def __str__(self):
        return f"#[{self.name}]"


class LambdaProcedure(Procedure):
    """
    A procedure defined by a lambda expression or a define form.
    """

    def __init__(self, formals, body, env):
        """
        A procedure with formal parameter list FORMALS (list),
        whose body is the Scheme list BODY, and whose parent environment
        starts with Frame ENV.
        """
        assert isinstance(env, Frame), "env must be of type Frame"

        self.formals = formals
        self.body = body
        self.env = env

    def __repr__(self):
        return "LambdaProcedure({0}, {1}, {2})".format(
            self.formals, self.body, self.env
        )


global_frame = create_global_frame()


##############
#    Pair    #
##############
def repl_str(val):
    """Should largely match str(val), except for booleans and undefined."""
    if val is True:
        return "#t"
    if val is False:
        return "#f"
    if val is None:
        return "undefined"
    if isinstance(val, str) and val and val[0] == '"':
        return '"' + repr(val[1:-1])[1:-1] + '"'
    return str(val)


class Pair:
    """
    A Pair has two instance attributes: car and cdr, cdr must be Pair or nil.
    """

    def __init__(self, first, rest):
        self.car = first
        self.cdr = rest

    def __repr__(self):
        return f"Pair({repr(self.car)}, {repr(self.cdr)})"

    def __str__(self):
        s = "(" + repl_str(self.car)
        rest = self.cdr
        while isinstance(rest, Pair):
            s += " " + repl_str(rest.car)
            rest = rest.cdr

        if rest is not nil:
            s += " . " + repl_str(rest)
        return s + ")"

    def __len__(self):
        n, rest = 1, self.cdr
        while isinstance(rest, Pair):
            n += 1
            rest = rest.cdr
        if rest is not nil:
            raise SchemeError("length attempted on improper list")
        return n

    def __eq__(self, p):
        if not isinstance(p, Pair):
            return False
        return self.car == p.car and self.cdr == p.cdr

    def map(self, fn):
        """Return a Scheme list after mapping Python function FN to SELF."""
        mapped = fn(self.first)
        if self.rest is nil or isinstance(self.rest, Pair):
            return Pair(mapped, self.rest.map(fn))
        else:
            raise TypeError("ill-formed list (cdr is a promise)")

    # def flatmap(self, fn):
    #     """Return a Scheme list after flatmapping Python function FN to SELF."""
    #     from scheme_builtins import scheme_append

    #     mapped = fn(self.first)
    #     if self.rest is nil or isinstance(self.rest, Pair):
    #         return scheme_append(mapped, self.rest.flatmap(fn))
    #     else:
    #         raise TypeError("ill-formed list (cdr is a promise)")


class nil:
    """The empty list"""

    def __repr__(self):
        return "nil"

    def __str__(self):
        return "[]"

    def __len__(self):
        return 0

    def map(self, fn):
        return self

    def flatmap(self, fn):
        return self


nil = nil()


##############
# Evaluation #
##############


# @trace
def evaluate_file(f_name, env=None):
    """
    Evaluate the syntax tree read from f_name in environment env.
    """
    with open(f_name, mode="r") as f:
        text = str(f.read())
        # print(f"got {text}")
        return evaluate(parse(tokenize(text)), env)


# @trace
def evaluate(tree, env=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """

    env = env if env is not None else make_initial_frame()
    tree = nil if isinstance(tree, list) and len(tree) == 0 else tree
    if scheme_symbolp(tree):
        return env.lookup(tree)
    elif self_evaluating(tree):
        return tree

    if not scheme_listp(tree):
        raise SchemeEvaluationError
    first, rest = tree[0], tree[1:]

    if scheme_symbolp(first) and first in scheme_special_forms:
        return scheme_special_forms[first](rest, env)
    else:
        proc = evaluate(first, env)
        return apply(proc, rest, env)


def make_function_frame(formals, args, env, func_env):
    """
    Evaluate args in environment ENV, and assign value to formals argument in func_env.
    """

    # (lambda (x) (+ x 1) 2)
    for name, expr in zip(formals, args):
        func_env.define(name, evaluate(expr, env))


def apply(procedure, args, env):
    """
    Apply Scheme PROCEDURE to argument value ARGS (list) in envrionment ENV.

    Arguments:
        procedure: an callable python function
        args: a list contain args of procedure
    """
    validate_procedure(procedure)
    if not isinstance(env, Frame):
        assert False, "Not a Frame: {}".format(env)

    if isinstance(procedure, BuiltinProcedure):
        if not procedure.need_env:
            args = list(map(lambda x: evaluate(x, env), args))
            return procedure.py_func(*args)
        else:
            return procedure.py_func(*args, env=env)

    elif isinstance(procedure, LambdaProcedure):
        func_frame = Frame(procedure.env)
        if len(args) != len(procedure.formals):
            raise SchemeEvaluationError(
                f"arguments numbers if not correct: {args}, but function formal argument is: {procedure.formals}"
            )
        make_function_frame(procedure.formals, args, env, func_frame)
        return evaluate(procedure.body, func_frame)
    else:
        assert False, "Unexpected procedure: {}".format(procedure)


if __name__ == "__main__":
    # NOTE THERE HAVE BEEN CHANGES TO THE REPL, KEEP THIS CODE BLOCK AS WELL
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    import os

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import schemerepl
    import sys

    global_frame = make_initial_frame()
    for f_name in sys.argv[1:]:
        evaluate_file(f_name, global_frame)
    schemerepl.SchemeREPL(
        sys.modules[__name__], use_frames=True, verbose=True, global_frame=global_frame
    ).cmdloop()
