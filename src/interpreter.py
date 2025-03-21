VERSION = "v0.1.0"
#######################################
# IMPORTS
#######################################

def string_with_arrows(text, pos_start, pos_end):
	result = ''

	# Calculate indices
	idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
	idx_end = text.find('\n', idx_start + 1)
	if idx_end < 0: idx_end = len(text)
	
	# Generate each line#
	line_count = pos_end.ln - pos_start.ln + 1
	for i in range(line_count):
		# Calculate line columns
		line = text[idx_start:idx_end]
		col_start = pos_start.col if i == 0 else 0
		col_end = pos_end.col if i == line_count - 1 else len(line) - 1

		# Append to result
		result += line + '\n'
		result += ' ' * col_start + '^' * (col_end - col_start)

		# Re-calculate indices
		idx_start = idx_end
		idx_end = text.find('\n', idx_start + 1)
		if idx_end < 0: idx_end = len(text)

	return result.replace('\t', '')

import string
import os
import time
import sys
import datetime

from enum import Enum, auto
from dataclasses import dataclass
from typing import *



files = {}



usePathReference = "src/path"
parent = os.getcwd()
import_paths = []

if not os.path.isfile(usePathReference):
    with open(usePathReference, "w") as pathFile:
        pathFile.write(os.getcwd() + "/std")
    import_paths.append(os.getcwd() + "/std")
else:
    with open(usePathReference, "r+") as pathFile:
        content = pathFile.read().strip()
        
        if not content:
            pathFile.seek(0)  
            pathFile.write(os.getcwd() + "/std")
            import_paths.append(os.getcwd() + "/std")
        elif "/std" not in content:
            pathFile.seek(0)  
            pathFile.write(os.getcwd() + "/std")
            import_paths.append(os.getcwd() + "/std")
        
        
        pathFile.seek(0)
        pathbulk = pathFile.read().splitlines()
        import_paths.extend(pathbulk)


DIGITS = '0123456789'
LETTERS = string.ascii_letters
VALID_IDENTIFIERS = LETTERS + DIGITS + "$_"


#######################################
# ERRORS
#######################################


class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def added_to(self, other):
        return None, self.illegal_operation(other)

    def subbed_by(self, other):
        return None, self.illegal_operation(other)

    def multed_by(self, other):
        return None, self.illegal_operation(other)

    def dived_by(self, other):
        return None, self.illegal_operation(other)

    def powed_by(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_lte(self, other):
        return None, self.illegal_operation(other)

    def get_comparison_gte(self, other):
        return None, self.illegal_operation(other)

    def anded_by(self, other):
        return None, self.illegal_operation(other)

    def ored_by(self, other):
        return None, self.illegal_operation(other)

    def notted(self):  # why need `other` for not?
        return None, self.illegal_operation()

    def iter(self):
        return Iterator(self.gen)

    def gen(self):
        yield RTResult().failure(self.illegal_operation())

    def get_index(self, index):
        return None, self.illegal_operation(index)

    def set_index(self, index, value):
        return None, self.illegal_operation(index, value)

    def execute(self, args):
        return RTResult().failure(self.illegal_operation())

    def get_dot(self, verb):
        t = type(self)
        attr = f"inner_{verb}"
        if not hasattr(t, attr):
            return None, RTError(
                self.pos_start, self.pos_end,
                f"Object of type '{t.__name__}' has no property of name '{verb}'",
                self.context
            )
        return getattr(t, attr), None

    def set_dot(self, verb, value):
        return None, self.illegal_operation(verb, value)

    def copy(self):
        raise Exception('No copy method defined')

    def is_true(self):
        return False

    def illegal_operation(self, *others):
        if len(others) == 0:
            others = self,

        return RTError(
            self.pos_start, others[-1].pos_end,
            'Illegal operation',
            self.context
        )


class Error(Value):
    def __init__(self, pos_start, pos_end, error_name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.error_name = error_name
        self.details = details

    def set_pos(self, pos_start=None, pos_end=None):
        return self

    def __repr__(self) -> str:
        return f'{self.error_name}: {self.details}'

    def as_string(self):
        result = f'{self.error_name}: {self.details}\n'
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += '\n\n' + \
            string_with_arrows(self.pos_start.ftxt,
                               self.pos_start, self.pos_end)
        return result

    def copy(self):
        return __class__(self.pos_start, self.pos_end, self.error_name, self.details)


class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)


class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details=''):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)


class RTError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.context = context

    def set_context(self, context=None):
        return self

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + \
            string_with_arrows(self.pos_start.ftxt,
                               self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result

    def copy(self):
        return __class__(self.pos_start, self.pos_end, self.details, self.context)


class TryError(RTError):
    def __init__(self, pos_start, pos_end, details, context, prev_error):
        super().__init__(pos_start, pos_end, details, context)
        self.prev_error = prev_error

    def generate_traceback(self):
        result = ""
        if self.prev_error:
            result += self.prev_error.as_string()
        result += "\nDuring the handling of the above error, another error occurred:\n\n"
        return result + super().generate_traceback()

#######################################
# POSITION
#######################################


class Position:
    def __init__(self, idx, ln, col, fn, ftxt):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.fn = fn
        self.ftxt = ftxt

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

    

#######################################
# TOKENS
#######################################


class TokenType(Enum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IDENTIFIER = auto()
    KEYWORD = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()
    EQ = auto()
    LPAREN = auto()
    RPAREN = auto()
    LSQUARE = auto()
    RSQUARE = auto()
    EE = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    COMMA = auto()
    ARROW = auto()
    LCURLY = auto()
    RCURLY = auto()
    COLON = auto()
    DOT = auto()
    NEWLINE = auto()
    EOF = auto()
    
    


KEYWORDS = [
    'and',
    'or',
    'not',
    'if',
    'elif',
    'else',
    'for',
    'to',
    'step',
    'while',
    'fn',
    'return',
    'end',
    'continue',
    'break',
    'use',
    'do',
    'try',
    'catch',
    'as',
    'from',
    'in',
    'switch',
    'case',
    'const',
    'namespace',

]


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end.copy()

    def copy(self):
        return Token(self.type, self.value, self.pos_start.copy(), self.pos_end.copy())

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value:
            return f'{self.type.name}:{self.value}'
        return f'{self.type.name}'

#######################################
# LEXER
#######################################


SINGLE_CHAR_TOKS: Dict[str, TokenType] = {
    ";": TokenType.NEWLINE,
    "\n": TokenType.NEWLINE,
    "+": TokenType.PLUS,
    "*": TokenType.MUL,
    "/": TokenType.DIV,
    "^": TokenType.POW,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LSQUARE,
    "]": TokenType.RSQUARE,
    "{": TokenType.LCURLY,
    "}": TokenType.RCURLY,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
    ".": TokenType.DOT
}


class Lexer:
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text
        self.pos = Position(-1, 0, -1, fn, text)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(
            self.text) else None
    

    def make_tokens(self):
        tokens = []

        while self.current_char != None:
            if self.current_char in SINGLE_CHAR_TOKS:
                tt = SINGLE_CHAR_TOKS[self.current_char]
                pos = self.pos.copy()
                self.advance()
                tokens.append(Token(tt, pos_start=pos))
            
            elif self.current_char.isspace():
                self.advance()
            elif self.current_char == '#':
                self.skip_comment()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char in VALID_IDENTIFIERS:
                tokens.append(self.make_identifier())
            elif self.current_char == '"':
                tokens.append(self.make_string())
            elif self.current_char == '-':
                tokens.append(self.make_minus_or_arrow())
            elif self.current_char == '!':
                token, error = self.make_not_equals()
                if error:
                    return [], error
                tokens.append(token)
            elif self.current_char == '=':
                tokens.append(self.make_equals())
            elif self.current_char == '<':
                tokens.append(self.make_less_than())
            elif self.current_char == '>':
                tokens.append(self.make_greater_than())
            elif self.current_char == '\\':
                self.advance()
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TokenType.EOF, pos_start=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
            num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TokenType.INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TokenType.FLOAT, float(num_str), pos_start, self.pos)

    def make_string(self):
        string = ''
        pos_start = self.pos.copy()
        escape_character = False
        self.advance()

        while self.current_char != None and (self.current_char != '"' or escape_character):
            if escape_character:
                escape_character = False
            elif self.current_char == '\\':
                escape_character = True
            string += self.current_char
            self.advance()

        self.advance()
        return Token(TokenType.STRING, string.encode('raw_unicode_escape').decode('unicode_escape'), pos_start, self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in VALID_IDENTIFIERS:
            id_str += self.current_char
            self.advance()

        tok_type = TokenType.KEYWORD if id_str in KEYWORDS else TokenType.IDENTIFIER
        return Token(tok_type, id_str, pos_start, self.pos)

    def make_minus_or_arrow(self):
        tok_type = TokenType.MINUS
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '>':
            self.advance()
            tok_type = TokenType.ARROW

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_not_equals(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            return Token(TokenType.NE, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")

    def make_equals(self):
        tok_type = TokenType.EQ
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TokenType.EE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        tok_type = TokenType.LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TokenType.LTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        
        tok_type = TokenType.GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == '=':
            self.advance()
            tok_type = TokenType.GTE

        return Token(tok_type, pos_start=pos_start, pos_end=self.pos)
    
    
    
    
    
    def skip_comment(self):
        multi_line_comment = False
        self.advance()
        if self.current_char == "*":
            multi_line_comment = True

        while self.current_char is not None:
            if self.current_char == "*" and multi_line_comment:
                self.advance()
                if self.current_char != "#":
                    continue
                else:
                    break
            elif self.current_char == "\n" and not multi_line_comment:
                break
            self.advance()

        self.advance()
    


class NumberNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class StringNode:
    def __init__(self, tok):
        self.tok = tok

        self.pos_start = self.tok.pos_start
        self.pos_end = self.tok.pos_end

    def __repr__(self):
        return f'{self.tok}'


class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.element_nodes = element_nodes

        self.pos_start = pos_start
        self.pos_end = pos_end


class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.var_name_tok.pos_end


class VarAssignNode:
    def __init__(self, var_name_tok, value_node, is_const=False):
        self.var_name_tok = var_name_tok
        self.value_node = value_node
        self.is_const = is_const

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.value_node.pos_end

    def __repr__(self):
        const = "const " if self.is_const else ""
        return f"({const}{self.var_name_tok} = {self.value_node!r})"


class BinOpNode:
    def __init__(self, left_node, op_tok, right_node):
        self.left_node = left_node
        self.op_tok = op_tok
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_tok}, {self.right_node})'


class UnaryOpNode:
    def __init__(self, op_tok, node):
        self.op_tok = op_tok
        self.node = node

        self.pos_start = self.op_tok.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op_tok}, {self.node})'


class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.pos_start = self.cases[0][0].pos_start
        self.pos_end = (
            self.else_case or self.cases[len(self.cases) - 1])[0].pos_end


class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.pos_start = self.var_name_tok.pos_start
        self.pos_end = self.body_node.pos_end


class WhileNode:
    def __init__(self, condition_node, body_node, should_return_null):
        self.condition_node = condition_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end


class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, defaults, dynamics, body_node, should_auto_return):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.defaults = defaults
        self.dynamics = dynamics
        self.body_node = body_node
        self.should_auto_return = should_auto_return

        if self.var_name_tok:
            self.pos_start = self.var_name_tok.pos_start
        elif len(self.arg_name_toks) > 0:
            self.pos_start = self.arg_name_toks[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start

        self.pos_end = self.body_node.pos_end


class CallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start

        if len(self.arg_nodes) > 0:
            self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end


class ReturnNode:
    def __init__(self, node_to_return, pos_start, pos_end):
        self.node_to_return = node_to_return

        self.pos_start = pos_start
        self.pos_end = pos_end


class ContinueNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


class BreakNode:
    def __init__(self, pos_start, pos_end):
        self.pos_start = pos_start
        self.pos_end = pos_end


@dataclass
class ImportNode:
    string_node: StringNode
    pos_start: Position
    pos_end: Position

    def __repr__(self) -> str:
        return f"use {self.string_node!r}"


@dataclass
class DoNode:
    statements: ListNode
    pos_start: Position
    pos_end: Position

    def __repr__(self) -> str:
        return f'(do {self.statements!r} end)'


@dataclass
class TryNode:
    try_block: ListNode
    exc_iden: Token
    catch_block: Any
    pos_start: Position
    pos_end: Position

    def __repr__(self) -> str:
        return f'(try {self.try_block!r} catch as {self.exc_iden!r} then {self.catch_block!r})'


@dataclass
class ForInNode:
    var_name_tok: Token
    iterable_node: Any
    body_node: Any
    pos_start: Position
    pos_end: Position
    should_return_null: bool

    def __repr__(self) -> str:
        return f"(for {self.var_name_tok} in {self.iterable_node!r} then {self.body_node!r})"


@dataclass
class IndexGetNode:
    indexee: Any
    index: Any
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        return f"({self.indexee!r}[{self.index!r}])"


@dataclass
class IndexSetNode:
    indexee: Any
    index: Any
    value: Any
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        return f"({self.indexee!r}[{self.index!r}]={self.value!r})"


@dataclass
class DictNode:
    pairs: Tuple[Any, Any]
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        result = "({"
        for key, value in self.pairs:
            result += f"{key!r}: {value!r}"
        return result + "})"


INDENTATION = 4


@dataclass
class SwitchNode:
    condition: Any
    cases: list[Tuple[Any, ListNode]]
    else_case: ListNode
    pos_start: Position
    pos_end: Position

    def __repr__(self):  
        return f"(switch {self.condition!r}\n " + (" " * INDENTATION) + ("\n " + " " * INDENTATION).join(
            f"case {case_cond!r}\n " + (" " * INDENTATION * 2) + f"{case_body!r}" for case_cond, case_body in list(self.cases)
        ) + "\n " + (" " * INDENTATION) + "else\n" + (" " * INDENTATION * 2) + f"{self.else_case!r})"


@dataclass
class DotGetNode:
    noun: Any
    verb: Token
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        return f"({self.noun!r}.{self.verb.value})"


@dataclass
class DotSetNode:
    noun: Any
    verb: Token
    value: Any
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        return f"({self.noun!r}.{self.verb.value}={self.value!r})"


@dataclass
class NamespaceNode:
    name: Optional[Token]
    body: Any
    pos_start: Position
    pos_end: Position

    def __repr__(self):
        return f"""(namespace {
      self.name.value if self.name is not None
      else ""
    }\n{self.body!r}\nEND)"""





#######################################
# PARSE RESULT
#######################################


class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.last_registered_advance_count = 0
        self.advance_count = 0
        self.to_reverse_count = 0

    def register_advancement(self):
        self.last_registered_advance_count = 1
        self.advance_count += 1

    def register(self, res):
        self.last_registered_advance_count = res.advance_count
        self.advance_count += res.advance_count
        if res.error:
            self.error = res.error
        return res.node

    def try_register(self, res):
        if res.error:
            self.to_reverse_count = res.advance_count
            return None
        return self.register(res)

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.last_registered_advance_count == 0:
            self.error = error
        return self

#######################################
# PARSER
#######################################


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        
        self.tok_idx = -1
        dummy = ParseResult()
        self.advance(dummy)

    def advance(self, res: ParseResult) -> Optional[Token]:
        self.tok_idx += 1
        self.update_current_tok()
        res.register_advancement()
        
        return self.current_tok
    
        

    def reverse(self, amount=1):
        self.tok_idx -= amount
        self.update_current_tok()
        return self.current_tok

    def update_current_tok(self):
        if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
            self.current_tok = self.tokens[self.tok_idx]

 
    def parse(self):
        res = self.statements()
        if not res.error and self.current_tok.type != TokenType.EOF:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Token cannot appear after previous tokens"
            ))
        return res

    ###################################

    def statements(self):
        res = ParseResult()
        statements = []
        pos_start = self.current_tok.pos_start.copy()

        while self.current_tok.type == TokenType.NEWLINE:
            self.advance(res)

        statement = res.register(self.statement())
        if res.error:
            return res
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0
            while self.current_tok.type == TokenType.NEWLINE:
                self.advance(res)
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements:
                break
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)

        return res.success(ListNode(
            statements,
            pos_start,
            self.current_tok.pos_end.copy()
        ))

    def statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()
        pos_end = self.current_tok.pos_end.copy()

        if self.current_tok.matches(TokenType.KEYWORD, 'return'):
            self.advance(res)

            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))

        if self.current_tok.matches(TokenType.KEYWORD, 'continue'):
            self.advance(res)
            return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))

        if self.current_tok.matches(TokenType.KEYWORD, 'break'):
            self.advance(res)
            return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))

        if self.current_tok.matches(TokenType.KEYWORD, 'use'):
            self.advance(res)

            if not self.current_tok.type == TokenType.STRING:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected string"
                ))

            string = res.register(self.atom())
            return res.success(ImportNode(string, pos_start, self.current_tok.pos_start.copy()))


        if self.current_tok.matches(TokenType.KEYWORD, 'try'):
            self.advance(res)
            try_node = res.register(self.try_statement())
            return res.success(try_node)

        if self.current_tok.matches(TokenType.KEYWORD, 'switch'):
            self.advance(res)
            switch_node = res.register(self.switch_statement())
            return res.success(switch_node)

        
        expr = res.register(self.expr())
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'switch', 'return', 'continue', 'break', 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
            ))
        return res.success(expr)

    def expr(self):
        res = ParseResult()

        var_assign_node = res.try_register(self.assign_expr())
        if var_assign_node:
            return res.success(var_assign_node)
        else:
            self.reverse(res.to_reverse_count)

        if self.current_tok.matches(TokenType.KEYWORD, "const"):
            self.advance(res)

            if self.current_tok.type != TokenType.IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected identifier"
                ))

            identifier = self.current_tok

            self.advance(res)

            if self.current_tok.type != TokenType.EQ:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '='"
                ))

            self.advance(res)

            assign_expr = res.register(self.expr())
            if res.error:
                return res

            return res.success(VarAssignNode(identifier, assign_expr, is_const=True))

        node = res.register(self.bin_op(
            self.comp_expr, ((TokenType.KEYWORD, 'and'), (TokenType.KEYWORD, 'or'))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
            ))

        if self.current_tok.type == TokenType.EQ:
            return res.failure(InvalidSyntaxError(
                node.pos_start, node.pos_end,
                "Invalid assignment"
            ))

        return res.success(node)

    def assign_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start

        if self.current_tok.type != TokenType.IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
            ))

        var_name_tok = self.current_tok

        self.advance(res)

        if self.current_tok.type != TokenType.EQ:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '='"
            ))

        self.advance(res)

        assign_expr = res.register(self.expr())
        if res.error:
            return res

        return res.success(VarAssignNode(var_name_tok, assign_expr))

    def comp_expr(self):
        res = ParseResult()

        if self.current_tok.matches(TokenType.KEYWORD, 'not'):
            op_tok = self.current_tok
            self.advance(res)

            node = res.register(self.comp_expr())
            if res.error:
                return res
            return res.success(UnaryOpNode(op_tok, node))

        node = res.register(self.bin_op(self.arith_expr, (TokenType.EE, TokenType.NE,
                            TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE)))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected int, float, identifier, '+', '-', '(', '[', 'if', 'for', 'while', 'fn' or 'not'"
            ))

        return res.success(node)

    def arith_expr(self):
        return self.bin_op(self.term, (TokenType.PLUS, TokenType.MINUS))

    def term(self):
        return self.bin_op(self.factor, (TokenType.MUL, TokenType.DIV))

    def factor(self):
        res = ParseResult()
        tok = self.current_tok

        if tok.type in (TokenType.PLUS, TokenType.MINUS):
            self.advance(res)
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()

    def power(self):
        return self.bin_op(self.call, (TokenType.POW, ), self.factor)

    def call(self):
        res = ParseResult()
        func = res.register(self.index())
        if res.error:
            return res

        if self.current_tok.type == TokenType.LPAREN:
            self.advance(res)
            arg_nodes = []

            if self.current_tok.type == TokenType.RPAREN:
                self.advance(res)
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected ')', 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
                    ))

                while self.current_tok.type == TokenType.COMMA:
                    self.advance(res)

                    arg_nodes.append(res.register(self.expr()))
                    if res.error:
                        return res

                if self.current_tok.type != TokenType.RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected ',' or ')'"
                    ))

                self.advance(res)
            return res.success(CallNode(func, arg_nodes))
        return res.success(func)

    def index(self):  # TODO: allow stuff like list[0].upper()[3].(...)
        res = ParseResult()
        noun = res.register(self.dot())
        if res.error:
            return res

        node = noun
        while self.current_tok.type == TokenType.LSQUARE:
            self.advance(res)
            index = res.register(self.expr())
            if res.error:
                return res

            if self.current_tok.type != TokenType.RSQUARE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_start,
                    "Expected ']'"
                ))

            node = IndexGetNode(node, index, node.pos_start,
                                self.current_tok.pos_end)
            self.advance(res)

        if self.current_tok.type == TokenType.EQ and isinstance(node, IndexGetNode):
            self.advance(res)

            value = res.register(self.expr())
            if res.error:
                return res

            node = IndexSetNode(node.indexee, node.index,
                                value, node.pos_start, self.current_tok.pos_end)

        return res.success(node)

    def dot(self):
        res = ParseResult()
        noun = res.register(self.atom())
        if res.error:
            return res

        node = noun
        while self.current_tok.type == TokenType.DOT:
            self.advance(res)

            if self.current_tok.type != TokenType.IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_start,
                    "Expected identifier"
                ))

            node = DotGetNode(node, self.current_tok,
                              node.pos_start, self.current_tok.pos_end)
            self.advance(res)

        if self.current_tok.type == TokenType.EQ and isinstance(node, DotGetNode):
            self.advance(res)

            value = res.register(self.expr())
            if res.error:
                return res

            node = DotSetNode(node.noun, node.verb, value,
                              node.pos_start, self.current_tok.pos_end)

        return res.success(node)

    def atom(self):
        res = ParseResult()
        tok = self.current_tok
        node = None

        if tok.type in (TokenType.INT, TokenType.FLOAT):
            self.advance(res)
            node = NumberNode(tok)

        elif tok.type == TokenType.STRING:
            self.advance(res)
            node = StringNode(tok)

        elif tok.type == TokenType.IDENTIFIER:
            self.advance(res)
            
            node = VarAccessNode(tok)

        elif tok.type == TokenType.LPAREN:
            self.advance(res)
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_tok.type == TokenType.RPAREN:
                self.advance(res)
                node = expr
            else:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ')'"
                ))

        elif tok.type == TokenType.LSQUARE:
            list_expr = res.register(self.list_expr())
            if res.error:
                return res
            node = list_expr

        elif tok.matches(TokenType.KEYWORD, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error:
                return res
            node = if_expr

        elif tok.matches(TokenType.KEYWORD, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error:
                return res
            node = for_expr

        elif tok.matches(TokenType.KEYWORD, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error:
                return res
            node = while_expr

        elif tok.matches(TokenType.KEYWORD, 'fn'):
            func_def = res.register(self.func_def())
            if res.error:
                return res
            node = func_def

        elif tok.matches(TokenType.KEYWORD, 'namespace'):
            ns_expr = res.register(self.namespace_expr())
            if res.error:
                return res
            node = ns_expr

        elif tok.matches(TokenType.KEYWORD, 'do'):
            do_expr = res.register(self.do_expr())
            if res.error:
                return res
            node = do_expr

        elif tok.type == TokenType.LCURLY:
            dict_expr = res.register(self.dict_expr())
            if res.error:
                return res
            node = dict_expr

        if node is None:
            return res.failure(InvalidSyntaxError(
                tok.pos_start, tok.pos_end,
                "Expected int, float, identifier, '+', '-', '(', '[', if', 'for', 'while', 'fn'"
            ))

        return res.success(node)

    def list_expr(self):
        res = ParseResult()
        element_nodes = []
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TokenType.LSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '['"
            ))

        self.advance(res)

        if self.current_tok.type == TokenType.RSQUARE:
            self.advance(res)
        else:
            element_nodes.append(res.register(self.expr()))
            if res.error:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ']', 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
                ))

            while self.current_tok.type == TokenType.COMMA:
                self.advance(res)

                element_nodes.append(res.register(self.expr()))
                if res.error:
                    return res

            if self.current_tok.type != TokenType.RSQUARE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ',' or ']'"
                ))

            self.advance(res)

        return res.success(ListNode(
            element_nodes,
            pos_start,
            self.current_tok.pos_end.copy()
        ))

    def dict_expr(self):
        res = ParseResult()
        pairs = []
        pos_start = self.current_tok.pos_start.copy()

        if self.current_tok.type != TokenType.LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        self.advance(res)

        if self.current_tok.type == TokenType.RCURLY:
            self.advance(res)
        else:
            key = res.register(self.expr())
            if res.error:
                return res

            if self.current_tok.type != TokenType.COLON:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ':'"
                ))

            self.advance(res)

            value = res.register(self.expr())
            if res.error:
                return res

            pairs.append((key, value))

            while self.current_tok.type == TokenType.COMMA:
                self.advance(res)

                key = res.register(self.expr())
                if res.error:
                    return res

                if self.current_tok.type != TokenType.COLON:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected ':'"
                    ))

                self.advance(res)

                value = res.register(self.expr())
                if res.error:
                    return res

                pairs.append((key, value))

            if self.current_tok.type != TokenType.RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ',' or '}'"
                ))

            self.advance(res)

        return res.success(DictNode(
            pairs,
            pos_start,
            self.current_tok.pos_end.copy()
        ))

    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_expr_cases('if'))
        if res.error:
            return res
        cases, else_case = all_cases
        return res.success(IfNode(cases, else_case))

    def if_expr_b(self):
        return self.if_expr_cases('elif')

    def if_expr_c(self):
        res = ParseResult()
        else_case = None

        if self.current_tok.matches(TokenType.KEYWORD, 'else'):
            self.advance(res)

            if self.current_tok.type == TokenType.NEWLINE:
                self.advance(res)

                statements = res.register(self.statements())
                if res.error:
                    return res
                else_case = (statements, True)

                if self.current_tok.type == TokenType.RCURLY:
                    self.advance(res)
                else:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected '}'"
                    ))
            else:
                expr = res.register(self.statement())
                if res.error:
                    return res
                else_case = (expr, False)

        return res.success(else_case)

    def if_expr_b_or_c(self):
        res = ParseResult()
        cases, else_case = [], None

        if self.current_tok.matches(TokenType.KEYWORD, 'elif'):
            all_cases = res.register(self.if_expr_b())
            if res.error:
                return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.if_expr_c())
            if res.error:
                return res
       
        return res.success((cases, else_case))

    def if_expr_cases(self, case_keyword):
        
        res = ParseResult()
        cases = []
        else_case = None
        
        
        if not self.current_tok.matches(TokenType.KEYWORD, case_keyword):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '{case_keyword}'"
            ))
        
        self.advance(res)

        condition = res.register(self.expr())
        
        if res.error:
            return res

        
        if self.current_tok.type != TokenType.LCURLY and case_keyword != "elif":

            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))
        if case_keyword != "elif":
            self.advance(res)

        if self.current_tok.type == TokenType.NEWLINE:
            
            self.advance(res)
            

            statements = res.register(self.statements())
            if res.error:
                return res
            cases.append((condition, statements, True))

            # Change "end" to "}"
            if self.current_tok.type == TokenType.RCURLY:  # Expecting '}'
                self.advance(res)
            else:
                all_cases = res.register(self.if_expr_b_or_c())
                if res.error:
                    return res
                new_cases, else_case = all_cases
                cases.extend(new_cases)
        else:
            expr = res.register(self.statement())
            if res.error:
                return res
            cases.append((condition, expr, False))

            all_cases = res.register(self.if_expr_b_or_c())
            if res.error:
                return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)

        return res.success((cases, else_case))


    def for_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        if not self.current_tok.matches(TokenType.KEYWORD, 'for'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'for'"
            ))

        self.advance(res)
        

        if self.current_tok.type != TokenType.IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected identifier"
            ))

        var_name = self.current_tok
        self.advance(res)

        is_for_in = False

        if self.current_tok.type != TokenType.EQ and not self.current_tok.matches(TokenType.KEYWORD, "in"):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected '=' or 'in'"
            ))
        elif self.current_tok.matches(TokenType.KEYWORD, "in"):
            self.advance(res)
            is_for_in = True

            iterable_node = res.register(self.expr())
            if res.error:
                return res

        else:
            self.advance(res)

            start_value = res.register(self.expr())
            if res.error:
                return res

            if not self.current_tok.matches(TokenType.KEYWORD, 'to'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected 'to'"
                ))

            self.advance(res)

            end_value = res.register(self.expr())
            if res.error:
                return res

            if self.current_tok.matches(TokenType.KEYWORD, 'step'):
                self.advance(res)

                step_value = res.register(self.expr())
                if res.error:
                    return res
            else:
                step_value = None

        if self.current_tok.type != TokenType.LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))

        self.advance(res)

        if self.current_tok.type == TokenType.NEWLINE:
            self.advance(res)

            body = res.register(self.statements())
            if res.error:
                return res

            if self.current_tok.type != TokenType.RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            pos_end = self.current_tok.pos_end.copy()
            self.advance(res)

            if is_for_in:
                return res.success(ForInNode(var_name, iterable_node, body, pos_start, pos_end, True))
            return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))

        body = res.register(self.statement())
        if res.error:
            return res

        pos_end = self.current_tok.pos_end.copy()

        if is_for_in:
            return res.success(ForInNode(var_name, iterable_node, body, pos_start, pos_end, False))
        return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

    def while_expr(self):
        res = ParseResult()

        if not self.current_tok.matches(TokenType.KEYWORD, 'while'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'while'"
            ))

        self.advance(res)

        condition = res.register(self.expr())
        if res.error:
            return res

        if self.current_tok.type != TokenType.LCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '{'"
            ))
        self.advance(res)

        if self.current_tok.type == TokenType.NEWLINE:
            self.advance(res)

            body = res.register(self.statements())
            if res.error:
                return res

            if self.current_tok.type != TokenType.RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}'"
                ))

            self.advance(res)

            return res.success(WhileNode(condition, body, True))

        body = res.register(self.statement())
        if res.error:
            return res

        return res.success(WhileNode(condition, body, False))

    def func_def(self):
        res = ParseResult()

        if not self.current_tok.matches(TokenType.KEYWORD, 'fn'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected 'fn'"
            ))

        self.advance(res)

        if self.current_tok.type == TokenType.IDENTIFIER:
            var_name_tok = self.current_tok
            self.advance(res)
            if self.current_tok.type != TokenType.LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected '('"
                ))
        else:
            var_name_tok = None
            if self.current_tok.type != TokenType.LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or '('"
                ))

        self.advance(res)
        arg_name_toks = []
        defaults = []
        dynamics = []
        hasOptionals = False

        if self.current_tok.type == TokenType.IDENTIFIER:
            pos_start = self.current_tok.pos_start.copy()
            pos_end = self.current_tok.pos_end.copy()
            arg_name_toks.append(self.current_tok)
            self.advance(res)

            if self.current_tok.type == TokenType.EQ:
                self.advance(res)
                default = res.register(self.expr())
                if res.error:
                    return res
                defaults.append(default)
                hasOptionals = True
            elif hasOptionals:
                return res.failure(InvalidSyntaxError(
                    pos_start, pos_end,
                    "Expected optional parameter."
                ))
            else:
                defaults.append(None)

            if self.current_tok.matches(TokenType.KEYWORD, 'from'):
                self.advance(res)
                dynamics.append(res.register(self.expr()))
                if res.error:
                    return res
            else:
                dynamics.append(None)

            while self.current_tok.type == TokenType.COMMA:
                self.advance(res)

                if self.current_tok.type != TokenType.IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        f"Expected identifier"
                    ))

                pos_start = self.current_tok.pos_start.copy()
                pos_end = self.current_tok.pos_end.copy()
                arg_name_toks.append(self.current_tok)
                self.advance(res)

                if self.current_tok.type == TokenType.EQ:
                    self.advance(res)
                    default = res.register(self.expr())
                    if res.error:
                        return res
                    defaults.append(default)
                    hasOptionals = True
                elif hasOptionals:
                    return res.failure(InvalidSyntaxError(
                        pos_start, pos_end,
                        "Expected optional parameter."
                    ))
                else:
                    defaults.append(None)

                if self.current_tok.matches(TokenType.KEYWORD, 'from'):
                    self.advance(res)
                    dynamics.append(res.register(self.expr()))
                    if res.error:
                        return res
                else:
                    dynamics.append(None)

            if self.current_tok.type != TokenType.RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected ',', ')' or '='"
                ))
        else:
            if self.current_tok.type != TokenType.RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    f"Expected identifier or ')'"
                ))

        self.advance(res)

        if self.current_tok.type == TokenType.ARROW:
            self.advance(res)

            body = res.register(self.expr())
            if res.error:
                return res

            return res.success(FuncDefNode(
                var_name_tok,
                arg_name_toks,
                defaults,
                dynamics,
                body,
                True
            ))
        

        if self.current_tok.type != TokenType.LCURLY:

            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '->' or {"
            ))

        self.advance(res)
        
        body = res.register(self.statements())
        
        if res.error:
            return res

        if self.current_tok.type != TokenType.RCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}'"
            ))

        self.advance(res)

        return res.success(FuncDefNode(
            var_name_tok,
            arg_name_toks,
            defaults,
            dynamics,
            body,
            False
        ))

    def do_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        self.advance(res)

        statements = res.register(self.statements())

        if self.current_tok.type != TokenType.RCURLY:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected '}', 'return', 'continue', 'break', 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
            ))

        pos_end = self.current_tok.pos_end.copy()
        self.advance(res)
        return res.success(DoNode(statements, pos_start, pos_end))

    def try_statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        try_block = res.register(self.statements())
        if res.error:
            return res

        if not self.current_tok.matches(TokenType.KEYWORD, 'catch'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'catch', 'return', 'continue', 'break', 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
            ))

        self.advance(res)

        if not self.current_tok.matches(TokenType.KEYWORD, 'as'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'as'"
            ))

        self.advance(res)

        if self.current_tok.type != TokenType.IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected identifier"
            ))

        exc_iden = self.current_tok.copy()
        self.advance(res)

        if self.current_tok.type != TokenType.NEWLINE:
            if self.current_tok.type != TokenType.LCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '{' or newline"
                ))

            self.advance(res)
            catch_block = res.register(self.statement())
        else:
            self.advance(res)
            catch_block = res.register(self.statements())
            if res.error:
                return res

            if self.current_tok != TokenType.RCURLY:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected '}', 'return', 'continue', 'break', 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
                ))

            self.advance(res)

        return res.success(TryNode(try_block, exc_iden, catch_block, pos_start, self.current_tok.pos_end.copy()))

    def switch_statement(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start

        condition = res.register(self.expr())
        if res.error:
            (res.error)
            return res

        if self.current_tok.type != TokenType.NEWLINE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected newline"
            ))
        self.advance(res)

        cases = []
        while self.current_tok.matches(TokenType.KEYWORD, "case"):
            self.advance(res)
            case = res.register(self.expr())
            if res.error:
                return res

            if self.current_tok.type != TokenType.NEWLINE:
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected newline"
                ))
            self.advance(res)

            body = res.register(self.statements())

            if res.error:
                return res

            cases.append((case, body))

        else_case = None
        if self.current_tok.matches(TokenType.KEYWORD, "else"):
            self.advance(res)
            else_case = res.register(self.statements())
            if res.error:
                return res

        if not self.current_tok.matches(TokenType.KEYWORD, "end"):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'end'"
            ))

        pos_end = self.current_tok.pos_end
        self.advance(res)

        node = SwitchNode(condition, cases, else_case, pos_start, pos_end)
        return res.success(node)



    def namespace_expr(self):
        res = ParseResult()
        pos_start = self.current_tok.pos_start.copy()

        self.advance(res)

        statements = res.register(self.statements())

        if not self.current_tok.matches(TokenType.KEYWORD, 'end'):
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected 'end', 'return', 'continue', 'break', 'if', 'for', 'while', 'fn', 'namespace', int, float, identifier, '+', '-', '(', '[', '{' or 'not'"
            ))

        pos_end = self.current_tok.pos_end.copy()
        self.advance(res)
        return res.success(DoNode(statements, pos_start, pos_end))

    ###################################

    def bin_op(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error:
            return res

        while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
            op_tok = self.current_tok
            self.advance(res)
            right = res.register(func_b())
            if res.error:
                return res
            left = BinOpNode(left, op_tok, right)

        return res.success(left)

#######################################
# RUNTIME RESULT
#######################################


class RTResult:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.loop_should_continue = False
        self.loop_should_break = False

    def register(self, res):
        self.error = res.error
        self.func_return_value = res.func_return_value
        self.loop_should_continue = res.loop_should_continue
        self.loop_should_break = res.loop_should_break
        return res.value

    def success(self, value):
        self.reset()
        self.value = value
        return self

    def success_return(self, value):
        self.reset()
        self.func_return_value = value
        return self

    def success_continue(self):
        self.reset()
        self.loop_should_continue = True
        return self

    def success_break(self):
        self.reset()
        self.loop_should_break = True
        return self

    def failure(self, error):
        self.reset()
        self.error = error
        return self

    def should_return(self):
        # Note: this will allow you to continue and break outside the current function
        return (
            self.error or
            self.func_return_value or
            self.loop_should_continue or
            self.loop_should_break
        )

#######################################
# VALUES
#######################################


class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def subbed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def dived_by(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Division by zero',
                    self.context
                )

            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def powed_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value != 0

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)


Number.null = Number(0)
Number.false = Number(0)
Number.true = Number(1)



class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def added_to(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def gen(self):
        for char in self.value:
            yield RTResult().success(String(char))

    def get_index(self, index):
        if not isinstance(index, Number):
            return None, self.illegal_operation(index)
        try:
            return self.value[index.value], None
        except IndexError:
            return None, RTError(
                index.pos_start, index.pos_end,
                f"Cannot retrieve character {index} from string {self!r} because it is out of bounds.",
                self.context
            )

    def get_comparison_eq(self, other):
        if not isinstance(other, String):
            return None, self.illegal_operation(other)
        return Number(int(self.value == other.value)), None

    def get_comparison_ne(self, other):
        if not isinstance(other, String):
            return None, self.illegal_operation(other)
        return Number(int(self.value != other.value)), None

    def is_true(self):
        return len(self.value) > 0

    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __str__(self):
        return self.value

    def __repr__(self):
        return f'"{self.value}"'
String.cwd = String(os.getcwd())
String._V = String(VERSION)

class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements
        self.value = elements

    def added_to(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None

    def subbed_by(self, other):
        if isinstance(other, Number):
            new_list = self.copy()
            try:
                new_list.elements.pop(other.value)
                return new_list, None
            except:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Element at this index could not be removed from list because index is out of bounds',
                    self.context
                )
        else:
            return None, Value.illegal_operation(self, other)

    def multed_by(self, other):
        if isinstance(other, List):
            new_list = self.copy()
            new_list.elements.extend(other.elements)
            return new_list, None
        else:
            return None, Value.illegal_operation(self, other)

    def dived_by(self, other):
        if isinstance(other, Number):
            try:
                return self.elements[other.value], None
            except:
                return None, RTError(
                    other.pos_start, other.pos_end,
                    'Element at this index could not be retrieved from list because index is out of bounds',
                    self.context
                )
        else:
            return None, Value.illegal_operation(self, other)

    def gen(self):
        for elt in self.elements:
            yield RTResult().success(elt)

    def get_index(self, index):
        if not isinstance(index, Number):
            return None, self.illegal_operation(index)
        try:
            return self.elements[index.value], None
        except IndexError:
            return None, RTError(
                index.pos_start, index.pos_end,
                f"Cannot retrieve element {index} from list {self!r} because it is out of bounds.",
                self.context
            )

    def set_index(self, index, value):
        if not isinstance(index, Number):
            return None, self.illegal_operation(index)
        try:
            self.elements[index.value] = value
        except IndexError:
            return None, RTError(
                index.pos_start, index.pos_end,
                f"Cannot set element {index} from list {self!r} to {value!r} because it is out of bounds.",
                self.context
            )

        return self, None

    def copy(self):
        copy = List(self.elements)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __str__(self):
        return ", ".join([str(x) for x in self.elements])

    def __repr__(self):
        return f'[{", ".join([repr(x) for x in self.elements])}]'


class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anonymous>"

    def set_context(self, context=None):
        if hasattr(self, "context") and self.context:
            return self
        return super().set_context(context)

    def generate_new_context(self):
        new_context = Context(self.name, self.context, self.pos_start)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
        return new_context

    def check_args(self, arg_names, args, defaults):
        res = RTResult()
        
        if len(args) > len(arg_names):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{len(args) - len(arg_names)} too many args passed into {self}",
                self.context
            ))

        if len(args) < len(arg_names) - len(list(filter(lambda default: default is not None, defaults))):
            return res.failure(RTError(
                self.pos_start, self.pos_end,
                f"{(len(arg_names) - len(list(filter(lambda default: default is not None, defaults)))) - len(args)} too few args passed into {self}",
                self.context
            ))

        return res.success(None)

    def populate_args(self, arg_names, args, defaults, dynamics, exec_ctx):
        res = RTResult()
        for i in range(len(arg_names)):
            arg_name = arg_names[i]
            dynamic = dynamics[i]
            arg_value = defaults[i] if i >= len(args) else args[i]
            if dynamic is not None:
                dynamic_context = Context(
                    f"{self.name} (dynamic argument '{arg_name}')", exec_ctx, dynamic.pos_start.copy())
                dynamic_context.symbol_table = SymbolTable(
                    exec_ctx.symbol_table)
                dynamic_context.symbol_table.set("$", arg_value)
                arg_value = res.register(
                    Interpreter().visit(dynamic, dynamic_context))
                if res.should_return():
                    return res
            arg_value.set_context(exec_ctx)
            exec_ctx.symbol_table.set(arg_name, arg_value)
        return res.success(None)

    def check_and_populate_args(self, arg_names, args, defaults, dynamics, exec_ctx):
        res = RTResult()
        res.register(self.check_args(arg_names, args, defaults))
        if res.should_return():
            return res
        res.register(self.populate_args(
            arg_names, args, defaults, dynamics, exec_ctx))
        if res.should_return():
            return res
        return res.success(None)


class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, defaults, dynamics, should_auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.defaults = defaults
        self.dynamics = dynamics
        self.should_auto_return = should_auto_return

    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        exec_ctx = self.generate_new_context()

        res.register(self.check_and_populate_args(self.arg_names,
                     args, self.defaults, self.dynamics, exec_ctx))
        if res.should_return():
            return res

        value = res.register(interpreter.visit(self.body_node, exec_ctx))
        if res.should_return() and res.func_return_value == None:
            return res

        ret_value = (
            value if self.should_auto_return else None) or res.func_return_value or Number.null
        return res.success(ret_value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names,
                        self.defaults, self.dynamics, self.should_auto_return)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"


class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)

    def execute(self, args):
        res = RTResult()
        exec_ctx = self.generate_new_context()

        method_name = f'execute_{self.name}'
        method = getattr(self, method_name, self.no_execute_method)

        res.register(self.check_and_populate_args(method.arg_names,
                     args, method.defaults, method.dynamics, exec_ctx))
        if res.should_return():
            return res

        return_value = res.register(method(exec_ctx))
        if res.should_return():
            return res
        return res.success(return_value)

    def no_execute_method(self, node, context):
        raise Exception(f'No execute_{self.name} method defined')

    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.pos_start, self.pos_end)
        return copy

    def __repr__(self):
        return f"<built-in function {self.name}>"

    #####################################

    # Decorator for built-in functions
    @staticmethod
    def args(arg_names, defaults=None, dynamics=None):
        
        if defaults is None:
            defaults = [None] * len(arg_names)
        if dynamics is None:
            dynamics = [None] * len(arg_names)
        
        
        def _args(f):

            f.arg_names = arg_names
            f.defaults = defaults
            f.dynamics = dynamics
        
            return f
        
        return _args

    #####################################

   

    @args(['value'])
    def execute_println(self, exec_ctx):
        print(str(exec_ctx.symbol_table.get('value')))
        return RTResult().success(Number.null)
    

    @args(['value'])
    def execute_system(self, exec_ctx):
        import subprocess
        command = str(exec_ctx.symbol_table.get('value'))  
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            return RTResult().success(String(result.stdout.strip()))  
        except Exception as e:
            return RTResult().failure(RuntimeError(f"System command failed: {str(e)}"))
    
    @args([])
    def execute_exit(self, exec_ctx = None):
        exit()
    
    
    @args(['value'])
    def execute_print(self, exec_ctx):
        print(str(exec_ctx.symbol_table.get('value')), end="")
        return RTResult().success(Number.null)
    

    @args(['mainString', 'splitBy'])
    def execute_split(self, exec_ctx):
      
      
        mainString = exec_ctx.symbol_table.get("mainString")
        splitBy = exec_ctx.symbol_table.get("splitBy")

        s2 = str(mainString).split(str(splitBy))
        try:
            for i, arg in enumerate(s2):
                s2[i] = String(arg)
            return RTResult().success(List(s2))
        except:
            return None
    
    @args(['mainString', 'suffix'])
    def execute_rmSuffix(self, exec_ctx):
        mainString = str(exec_ctx.symbol_table.get("mainString"))
        suffix = str(exec_ctx.symbol_table.get("suffix"))
        text = mainString.removesuffix(suffix)
        return RTResult().success(String(text))
    
    @args(['mainString', 'prefix'])
    def execute_rmPrefix(self, exec_ctx):
        mainString = str(exec_ctx.symbol_table.get("mainString"))
        prefix = str(exec_ctx.symbol_table.get("prefix"))
        text = mainString.removeprefix(prefix)
        return RTResult().success(String(text))
    
    

    @args(['mainString', 'value'], defaults=[None, String(" ")])
    def execute_clean(self, exec_ctx):
        mainString = exec_ctx.symbol_table.get("mainString")
        if not exec_ctx.symbol_table.get("value"):
            text = mainString.strip()
        else:
            text = str(mainString).strip(str(exec_ctx.symbol_table.get("value")))
        return RTResult().success(String(text))

    @args(['mainString'])
    def execute_stack(self, exec_ctx):
        mainString = str(exec_ctx.symbol_table.get("mainString"))
        text = mainString.splitlines()
        for i, line in enumerate(text):
            text[i] = String(line)
        return RTResult().success(List(text))
    

    @args(['mainString', 'seek', 'replace'])
    def execute_swapOut(self, exec_ctx):
        mainString = str(exec_ctx.symbol_table.get("mainString"))
        seek = str(exec_ctx.symbol_table.get("seek"))
        replace = str(exec_ctx.symbol_table.get("replace"))
        text = mainString.replace(seek, replace)
        return RTResult().success(String(text))

    @args(["value"], defaults=[String("")])
    def execute_read(self, exec_ctx):
        prompt = str(exec_ctx.symbol_table.get('value'))
        if prompt.strip() == "":
            text = input()
        else:
            text = input(prompt)
        return RTResult().success(String(text))
    


    @args(["value"])
    def execute_str(self, exec_ctx):
        text = str(exec_ctx.symbol_table.get('value'))
        return RTResult().success(String(text))
    
    @args(["value"])
    def execute_int(self, exec_ctx):
        text = int(str(exec_ctx.symbol_table.get('value')))
        return RTResult().success(Number(text))

    @args(["value"])
    def execute_flt(self, exec_ctx):
        text = float(str(exec_ctx.symbol_table.get('value')))
        return RTResult().success(Number(text))

    @args(["value"])
    def execute_list(self, exec_ctx):
        text = list(str(exec_ctx.symbol_table.get('value')))
        return RTResult().success(Number(text))
    

    @args([])
    def execute_clear(self, exec_ctx):
        os.system('cls' if os.name == 'nt' else 'cls')
        return RTResult().success(Number.null)
    
    @args([])
    def execute_time(self, exec_ctx):
        current_time = datetime.datetime.now().time()  # Gets the current time
        return RTResult().success(String(str(current_time)))  # Return as a string

    @args([])
    def execute_date(self, exec_ctx):
        current_date = datetime.date.today()  # Gets today's date
        return RTResult().success(String(str(current_date)))

    @args(["value"])
    def execute_isNum(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
        return RTResult().success(Number.true if is_number else Number.false)

    @args(["value"])
    def execute_isStr(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
        return RTResult().success(Number.true if is_number else Number.false)

    @args(["value"])
    def execute_isList(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
        return RTResult().success(Number.true if is_number else Number.false)

    @args(["value"])
    def execute_isFn(self, exec_ctx):
        is_number = isinstance(
            exec_ctx.symbol_table.get("value"), BaseFunction)
        return RTResult().success(Number.true if is_number else Number.false)

    @args(["list", "value"])
    def execute_append(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")
        value = exec_ctx.symbol_table.get("value")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be list",
                exec_ctx
            ))

        list_.elements.append(value)
        return RTResult().success(Number.null)

    @args(["list", "index"])
    def execute_pop(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")
        index = exec_ctx.symbol_table.get("index")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be list",
                exec_ctx
            ))

        if not isinstance(index, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be number",
                exec_ctx
            ))

        try:
            element = list_.elements.pop(index.value)
        except:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                'Element at this index could not be removed from list because index is out of bounds',
                exec_ctx
            ))
        return RTResult().success(element)

    @args(["listA", "listB"])
    def execute_extend(self, exec_ctx):
        listA = exec_ctx.symbol_table.get("listA")
        listB = exec_ctx.symbol_table.get("listB")

        if not isinstance(listA, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be list",
                exec_ctx
            ))

        if not isinstance(listB, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be list",
                exec_ctx
            ))

        listA.elements.extend(listB.elements)
        return RTResult().success(Number.null)

    @args(["list"])
    def execute_len(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be list",
                exec_ctx
            ))

        return RTResult().success(Number(len(list_.elements)))

    @args(["fn"])
    def execute_run(self, exec_ctx):
        fn = exec_ctx.symbol_table.get("fn")

        if not isinstance(fn, String):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Second argument must be string",
                exec_ctx
            ))

        fn = fn.value

        try:
            with open(fn, "r") as f:
                script = f.read()
        except Exception as e:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Failed to load script \"{fn}\"\n" + str(e),
                exec_ctx
            ))

        _, error = run(fn, script)

        if error:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                f"Failed to finish executing script \"{fn}\"\n" +
                error.as_string(),
                exec_ctx
            ))

        return RTResult().success(Number.null)

    @args(["fn", "mode"], [None, String("r")])
    def execute_openFile(self, exec_ctx):
        sym = exec_ctx.symbol_table
        fake_pos = create_fake_pos("<built-in function open>")
        res = RTResult()

        fn = sym.get("fn")
        if not isinstance(fn, String):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"1st argument of function 'open' ('fn') must be String",
                exec_ctx
            ))
        fn = fn.value

        mode = sym.get("mode")
        if not isinstance(mode, String):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"2nd argument of function 'open' ('mode') must be String",
                exec_ctx
            ))
        mode = mode.value

        try:
            f = open(fn, mode)
        except (TypeError, OSError) as err:
            if isinstance(err, TypeError):
                return res.failure(RTError(
                    fake_pos, fake_pos,
                    f"Invalid file open mode: '{mode}'",
                    exec_ctx
                ))
            elif isinstance(err, FileNotFoundError):
                return res.failure(RTError(
                    fake_pos, fake_pos,
                    f"Cannot find file '{fn}'",
                    exec_ctx
                ))
            else:
                return res.failure(RTError(
                    fake_pos, fake_pos,
                    f"{err.args[-1]}",
                    exec_ctx
                )) 

        fd = f.fileno()
        files[fd] = f

        return res.success(Number(fd).set_pos(fake_pos, fake_pos).set_context(exec_ctx))

    @args(["fd", "bytes"])
    def execute_readFile(self, exec_ctx):
        sym = exec_ctx.symbol_table
        fake_pos = create_fake_pos("<built-in function read>")
        res = RTResult()

        fd = sym.get("fd")
        if not isinstance(fd, Number):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"1st argument of function 'read' ('fd') must be Number",
                exec_ctx
            ))
        fd = fd.value

        bts = sym.get("bytes")
        if not isinstance(bts, Number):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"2nd argument of function 'read' ('bytes') must be Number",
                exec_ctx
            ))
        bts = bts.value

        try:
            result = os.read(fd, bts).decode("utf-8")
        except OSError:
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"Invalid file descriptor: {fd}",
                exec_ctx
            ))

        return res.success(String(result).set_pos(fake_pos, fake_pos).set_context(exec_ctx))

    @args(["fd", "bytes"])
    def execute_writeFile(self, exec_ctx):
        sym = exec_ctx.symbol_table
        fake_pos = create_fake_pos("<built-in function writeFile>")
        res = RTResult()

        fd = sym.get("fd")
        if not isinstance(fd, Number):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"1st argument of function 'write' ('fd') must be Number",
                exec_ctx
            ))
        fd = fd.value

        bts = sym.get("bytes")
        if not isinstance(bts, String):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"2nd argument of function 'write' ('bytes') must be String",
                exec_ctx
            ))
        bts = bts.value

        try:
            num = os.write(fd, bytes(bts, "utf-8"))
        except OSError:
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"Invalid file descriptor: {fd}",
                exec_ctx
            ))

        return res.success(Number(num).set_pos(fake_pos, fake_pos).set_context(exec_ctx))

    @args(["fd"])
    def execute_closeFile(self, exec_ctx):
        sym = exec_ctx.symbol_table
        fake_pos = create_fake_pos("<built-in function close>")
        res = RTResult()

        fd = sym.get("fd")
        if not isinstance(fd, Number):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"1st argument of function 'close' ('fd') must be Number",
                exec_ctx
            ))
        fd = fd.value
        std_desc = ["stdin", "stdout", "stderr"]

        if fd < 3:
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"Cannot close {std_desc[fd]}",
                exec_ctx
            ))

        try:
            os.close(fd)
            del fd
        except OSError:
            pass

        return res.success(Number.null)

    @args(["secs"])
    def execute_wait(self, exec_ctx):
        sym = exec_ctx.symbol_table
        fake_pos = create_fake_pos("<built-in function wait>")
        res = RTResult()

        secs = sym.get("secs")
        if not isinstance(secs, Number):
            return res.failure(RTError(
                fake_pos, fake_pos,
                f"1st argument of function 'wait' ('secs') must be Number",
                exec_ctx
            ))
        secs = secs.value

        time.sleep(secs)

        return RTResult().success(Number.null)


BuiltInFunction.println = BuiltInFunction("println")
BuiltInFunction.debug = BuiltInFunction("debug")

BuiltInFunction.print = BuiltInFunction("print")

BuiltInFunction.date = BuiltInFunction("date")
BuiltInFunction.time = BuiltInFunction("time")

BuiltInFunction.system = BuiltInFunction("system")
BuiltInFunction.exit = BuiltInFunction("exit")



BuiltInFunction.read = BuiltInFunction("read")
BuiltInFunction.clear = BuiltInFunction("clear")
BuiltInFunction.isNum = BuiltInFunction("isNum")
BuiltInFunction.isStr = BuiltInFunction("isStr")
BuiltInFunction.isList = BuiltInFunction("isList")
BuiltInFunction.isFn = BuiltInFunction("isFn")

BuiltInFunction.str = BuiltInFunction("str")
BuiltInFunction.int = BuiltInFunction("int")
BuiltInFunction.flt = BuiltInFunction("flt")
BuiltInFunction.list = BuiltInFunction("list")

BuiltInFunction.split = BuiltInFunction("split")
BuiltInFunction.stack = BuiltInFunction("stack")
BuiltInFunction.clean = BuiltInFunction("clean")
BuiltInFunction.swapOut = BuiltInFunction("swapOut")
BuiltInFunction.rmSuffix = BuiltInFunction("rmSuffix")
BuiltInFunction.rmPrefix = BuiltInFunction("rmPrefix")



BuiltInFunction.append = BuiltInFunction("append")
BuiltInFunction.pop = BuiltInFunction("pop")
BuiltInFunction.extend = BuiltInFunction("extend")
BuiltInFunction.len = BuiltInFunction("len")
BuiltInFunction.run = BuiltInFunction("run")
BuiltInFunction.openFile = BuiltInFunction("openFile")


BuiltInFunction.readFile = BuiltInFunction("readFile")
BuiltInFunction.writeFile = BuiltInFunction("writeFile")
BuiltInFunction.closeFile = BuiltInFunction("closeFile")
BuiltInFunction.wait = BuiltInFunction("wait")


class Iterator(Value):
    def __init__(self, generator):
        super().__init__()
        self.it = generator()

    def iter(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.it)

    def __str__(self):
        return '<iterator>'

    def __repr__(self):
        return str(self)

    def __getattr__(self, attr):
        if attr.startswith("get_comparison_"):
            return lambda self, other: Number(self is other), None

    def copy(self):
        return Iterator(self.it)


class Dict(Value):
    def __init__(self, values):
        super().__init__()
        self.values = values
        self.value = values

    def added_to(self, other):
        if not isinstance(other, Dict):
            return None, self.illegal_operation(other)

        new_dict = self.copy()
        for key, value in other.values.items():
            new_dict.values[key] = value

        return new_dict, None

    def gen(self):
        fake_pos = create_fake_pos("<dict key>")
        for key in self.values.keys():
            key_as_value = String(key).set_pos(
                fake_pos, fake_pos).set_context(self.context)
            yield RTResult().success(key_as_value)

    def get_index(self, index):
        if not isinstance(index, String):
            return None, self.illegal_operation(index)

        try:
            return self.values[index.value], None
        except KeyError:
            return None, RTError(
                self.pos_start, self.pos_end,
                f"Could not find key {index!r} in dict {self!r}",
                self.context
            )

    def set_index(self, index, value):
        if not isinstance(index, String):
            return None, self.illegal_operation(index)

        self.values[index.value] = value

        return self, None

    def __str__(self):
        result = ""
        for key, value in self.values.items():
            result += f"{key}: {value}\n"

        return result[:-1]

    def __repr__(self):
        result = "{"
        for key, value in self.values.items():
            result += f"{key!r}: {value!r}, "

        return result[:-2] + "}"

    def copy(self):
        return Dict(self.values).set_pos(self.pos_start, self.pos_end).set_context(self.context)




#######################################
# CONTEXT
#######################################


class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

#######################################
# SYMBOL TABLE
#######################################


class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.structs = {}
        self.parent = parent
        self.const = set()

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def set_const(self, name, value):
        self.symbols[name] = value
        self.const.add(name)

    def remove(self, name):
        del self.symbols[name]

#######################################
# INTERPRETER
#######################################


class Interpreter:
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    ###################################

    def visit_NumberNode(self, node, context):
        return RTResult().success(
            Number(node.tok.value).set_context(
                context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_StringNode(self, node, context):
        return RTResult().success(
            String(node.tok.value).set_context(
                context).set_pos(node.pos_start, node.pos_end)
        )

    def visit_ListNode(self, node, context):
        res = RTResult()
        elements = []

        for element_node in node.element_nodes:
            elements.append(res.register(self.visit(element_node, context)))
            if res.should_return():
                return res

        return res.success(
            List(elements).set_context(context).set_pos(
                node.pos_start, node.pos_end)
        )

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if not value:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"'{var_name}' is not defined",
                context
            ))

        value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = res.register(self.visit(node.value_node, context))
        if res.should_return():
            return res

        if node.is_const:
            method = context.symbol_table.set_const
        else:
            method = context.symbol_table.set

        if var_name not in context.symbol_table.const:
            method(var_name, value)
            return res.success(value)
        else:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"Assignment to constant variable '{var_name}'",
                context
            ))

    def visit_BinOpNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.should_return():
            return res
        right = res.register(self.visit(node.right_node, context))
        if res.should_return():
            return res

        if node.op_tok.type == TokenType.PLUS:
            result, error = left.added_to(right)
        elif node.op_tok.type == TokenType.MINUS:
            result, error = left.subbed_by(right)
        elif node.op_tok.type == TokenType.MUL:
            result, error = left.multed_by(right)
        elif node.op_tok.type == TokenType.DIV:
            result, error = left.dived_by(right)
        elif node.op_tok.type == TokenType.POW:
            result, error = left.powed_by(right)
        elif node.op_tok.type == TokenType.EE:
            result, error = left.get_comparison_eq(right)
        elif node.op_tok.type == TokenType.NE:
            result, error = left.get_comparison_ne(right)
        elif node.op_tok.type == TokenType.LT:
            result, error = left.get_comparison_lt(right)
        elif node.op_tok.type == TokenType.GT:
            result, error = left.get_comparison_gt(right)
        elif node.op_tok.type == TokenType.LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op_tok.type == TokenType.GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op_tok.matches(TokenType.KEYWORD, 'and'):
            result, error = left.anded_by(right)
        elif node.op_tok.matches(TokenType.KEYWORD, 'or'):
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RTResult()
        number = res.register(self.visit(node.node, context))
        if res.should_return():
            return res

        error = None

        if node.op_tok.type == TokenType.MINUS:
            number, error = number.multed_by(Number(-1))
        elif node.op_tok.matches(TokenType.KEYWORD, 'not'):
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr, should_return_null in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.should_return():
                return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.should_return():
                    return res
                return res.success(Number.null if should_return_null else expr_value)

        if node.else_case:
            expr, should_return_null = node.else_case
            expr_value = res.register(self.visit(expr, context))
            if res.should_return():
                return res
            return res.success(Number.null if should_return_null else expr_value)

        return res.success(Number.null)

    def visit_ForNode(self, node, context):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return():
            return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return():
            return res

        if node.step_value_node:
            step_value = res.register(
                self.visit(node.step_value_node, context))
            if res.should_return():
                return res
        else:
            step_value = Number(1)

        i = start_value.value

        if step_value.value >= 0:
            def condition(): return i < end_value.value
        else:
            def condition(): return i > end_value.value

        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False:
                return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(
                node.pos_start, node.pos_end)
        )

    def visit_WhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return():
                return res

            if not condition.is_true():
                break

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False:
                return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(
                node.pos_start, node.pos_end)
        )

    def visit_FuncDefNode(self, node, context):
        res = RTResult()

        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        defaults = []
        for default in node.defaults:
            if default is None:
                defaults.append(None)
                continue
            default_value = res.register(self.visit(default, context))
            if res.should_return():
                return res
            defaults.append(default_value)

        func_value = Function(func_name, body_node, arg_names, defaults, node.dynamics,
                              node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)

        if node.var_name_tok:
            context.symbol_table.set(func_name, func_value)

        return res.success(func_value)

    def visit_CallNode(self, node, context):
        res = RTResult()
        args = []

        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.should_return():
            return res
        value_to_call = value_to_call.copy().set_pos(node.pos_start, node.pos_end)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.should_return():
                return res

        return_value = res.register(value_to_call.execute(args))
        if res.should_return():
            return res
        return_value = return_value.copy().set_pos(
            node.pos_start, node.pos_end).set_context(context)
        return res.success(return_value)

    def visit_ReturnNode(self, node, context):
        res = RTResult()

        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, context))
            if res.should_return():
                return res
        else:
            value = Number.null

        return res.success_return(value)

    def visit_ContinueNode(self, node, context):
        return RTResult().success_continue()

    def visit_BreakNode(self, node, context):
        return RTResult().success_break()

    def visit_ImportNode(self, node, context):
        res = RTResult()
        filename = res.register(self.visit(node.string_node, context))
        formatted = str(filename)
        code = None
        if "/" in formatted:
            if formatted.endswith("/") == False:
                formatted = formatted.split("/")
                lastarg = formatted[len(formatted)-1]
                if "." not in lastarg:
                    filename.value = filename.value + ".tr"
        else:
            if "." not in formatted:
                filename.value = filename.value + ".tr"
                

        for path in import_paths:
            if path.startswith("."):
                path = path.replace(".", parent)
             
                
                
            try:
                filepath = os.path.join(path, filename.value)
                with open(filepath, "r") as f:
                    code = f.read()
                    beginning = "/" if filepath.startswith("/") else ""
                    split = filepath.split("/")
                    split = beginning + "/".join(split[:-1]), split[-1]
                    os.chdir(split[0])
                    filename = split[1]
                    break
            except FileNotFoundError:
                continue

        if code is None:
            return res.failure(RTError(
                node.string_node.pos_start.copy(), node.string_node.pos_end.copy(),
                f"Can't find file '{filepath}' in '{usePathReference}'. Please add the directory your file is into that file",
                context
            ))

        _, error = run(filename, code, context, node.pos_start.copy())
        if error:
            return res.failure(error)

        return res.success(Number.null)

    def visit_DoNode(self, node, context):
        res = RTResult()
        new_context = Context("<do statement>", context, node.pos_start.copy())
        new_context.symbol_table = SymbolTable(context.symbol_table)
        res.register(self.visit(node.statements, new_context))

        return_value = res.func_return_value
        if res.should_return() and return_value is None:
            return res

        return_value = return_value or Number.null

        return res.success(return_value)

    def visit_TryNode(self, node: TryNode, context):
        res = RTResult()
        res.register(self.visit(node.try_block, context))
        handled_error = res.error
        if res.should_return() and res.error is None:
            return res

        elif handled_error is not None:
            var_name = node.exc_iden.value
            context.symbol_table.set(var_name, res.error)
            res.error = None

            res.register(self.visit(node.catch_block, context))
            if res.error:
                return res.failure(TryError(
                    res.error.pos_start, res.error.pos_end, res.error.details, res.error.context, handled_error
                ))
            return res.success(Number.null)
        else:
            return res.success(Number.null)

    def visit_ForInNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        body = node.body_node
        should_return_null = node.should_return_null

        iterable = res.register(self.visit(node.iterable_node, context))
        it = iterable.iter()

        elements = []

        for it_res in it:
            elt = res.register(it_res)
            if res.should_return():
                return res

            context.symbol_table.set(var_name, elt)

            elements.append(res.register(self.visit(body, context)))
            if res.should_return():
                return res

        if should_return_null:
            return res.success(Number.null)
        return res.success(elements)

    def visit_IndexGetNode(self, node, context):
        res = RTResult()
        indexee = res.register(self.visit(node.indexee, context))
        if res.should_return():
            return res

        index = res.register(self.visit(node.index, context))
        if res.should_return():
            return res

        result, error = indexee.get_index(index)
        if error:
            return res.failure(error)
        return res.success(result)

    def visit_IndexSetNode(self, node, context):
        res = RTResult()
        indexee = res.register(self.visit(node.indexee, context))
        if res.should_return():
            return res

        index = res.register(self.visit(node.index, context))
        if res.should_return():
            return res

        value = res.register(self.visit(node.value, context))
        if res.should_return():
            return res

        result, error = indexee.set_index(index, value)
        if error:
            return res.failure(error)

        return res.success(result)

    def visit_DictNode(self, node, context):
        res = RTResult()
        values = {}

        for key_node, value_node in node.pairs:
            key = res.register(self.visit(key_node, context))
            if res.should_return():
                return res

            if not isinstance(key, String):
                return res.failure(RTError(
                    key_node.pos_start, key_node.pos_end,
                    f"Non-string key for dict: '{key!r}'",
                    context
                ))

            value = res.register(self.visit(value_node, context))
            if res.should_return():
                return res

            values[key.value] = value

        return res.success(Dict(values))

    def visit_SwitchNode(self, node, context):
        res = RTResult()
        condition = res.register(self.visit(node.condition, context))
        if res.should_return():
            return res

        for case, body in node.cases:
            case = res.register(self.visit(case, context))
            if res.should_return():
                return res

            # print(f"[DEBUG] {object.__repr__(case)}")

            eq, error = condition.get_comparison_eq(case)
            if error:
                return res.failure(error)

            if eq.value:
                res.register(self.visit(body, context))
                if res.should_return():
                    return res
                break
        else:  # no break
            else_case = node.else_case
            if else_case:
                res.register(self.visit(else_case, context))
                if res.should_return():
                    return res

        return res.success(Number.null)

    def visit_DotGetNode(self, node, context):
        res = RTResult()
        noun = res.register(self.visit(node.noun, context))
        if res.should_return():
            return res

        verb = node.verb.value

        result, error = noun.get_dot(verb)
        if error:
            return res.failure(error)
        return res.success(result)

    def visit_DotSetNode(self, node, context):
        res = RTResult()
        noun = res.register(self.visit(node.noun, context))
        if res.should_return():
            return res

        verb = node.verb.value

        value = res.register(self.visit(node.value, context))
        if res.should_return():
            return res

        result, error = noun.set_dot(verb, value)
        if error:
            return res.failure(error)

        return res.success(result)

    

#######################################
# CREATE FAKE POS
#######################################


def create_fake_pos(desc: str) -> Position:
    return Position(0, 0, 0, desc, "<native code>")  # hmm yes very native

#######################################
# RUN
#######################################


def make_argv():
    argv = []
    fake_pos = create_fake_pos("<argv>")
    for arg in sys.argv[1:]:
        argv.append(String(arg).set_pos(fake_pos, fake_pos))
    return List(argv).set_pos(fake_pos, fake_pos)


global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number.null)
global_symbol_table.set("false", Number.false)
global_symbol_table.set("true", Number.true)
global_symbol_table.set("cwd", String.cwd)
global_symbol_table.set("_V", String._V)

global_symbol_table.set("argv", make_argv())
global_symbol_table.set("println", BuiltInFunction.println)

global_symbol_table.set("exit", BuiltInFunction.exit)
global_symbol_table.set("system", BuiltInFunction.system)


global_symbol_table.set("print", BuiltInFunction.print)

global_symbol_table.set("date", BuiltInFunction.time)
global_symbol_table.set("time", BuiltInFunction.time)


global_symbol_table.set("read", BuiltInFunction.read)
global_symbol_table.set("clear", BuiltInFunction.clear)
global_symbol_table.set("isNum", BuiltInFunction.isNum)
global_symbol_table.set("isStr", BuiltInFunction.isStr)

global_symbol_table.set("str", BuiltInFunction.str)
global_symbol_table.set("int", BuiltInFunction.int)
global_symbol_table.set("flt", BuiltInFunction.flt)
global_symbol_table.set("list", BuiltInFunction.list)


global_symbol_table.set("split", BuiltInFunction.split)
global_symbol_table.set("stack", BuiltInFunction.stack)
global_symbol_table.set("clean", BuiltInFunction.clean)
global_symbol_table.set("swapOut", BuiltInFunction.swapOut)
global_symbol_table.set("rmPrefix", BuiltInFunction.rmPrefix)
global_symbol_table.set("rmSuffix", BuiltInFunction.rmSuffix)


global_symbol_table.set("isList", BuiltInFunction.isList)
global_symbol_table.set("isFn", BuiltInFunction.isFn)
global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("len", BuiltInFunction.len)
global_symbol_table.set("run", BuiltInFunction.run)
global_symbol_table.set("openFile", BuiltInFunction.openFile)
global_symbol_table.set("readFile", BuiltInFunction.readFile)
global_symbol_table.set("writeFile", BuiltInFunction.writeFile)
global_symbol_table.set("closeFile", BuiltInFunction.closeFile)
global_symbol_table.set("wait", BuiltInFunction.wait)


def run(fn, text, context=None, entry_pos=None):
    # Generate tokens
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_tokens()
    if error:
        return None, error

    # Generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error:
        return None, ast.error

    # Run program
    interpreter = Interpreter()
    context_was_none = context is None
    context = Context('<program>', context, entry_pos)
    if context_was_none:
        context.symbol_table = global_symbol_table
    else:
        context.symbol_table = context.parent.symbol_table
    result = interpreter.visit(ast.node, context)
    ret = result.func_return_value
    if context_was_none and ret:
        if not isinstance(ret, Number):
            return None, RTError(
                ret.pos_start, ret.pos_end,
                "Exit code must be Number",
                context
            )
        exit(ret.value)

    return result.value, result.error
