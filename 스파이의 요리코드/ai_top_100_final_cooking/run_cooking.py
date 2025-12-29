# coding: utf-8
import argparse
import math
import re
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


class EvalError(Exception):
    pass


class UnknownCondition(EvalError):
    pass


class Expr:
    def eval_int(self) -> Optional[int]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Const(Expr):
    value: int

    def eval_int(self) -> Optional[int]:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def eval_int(self) -> Optional[int]:
        return None

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr

    def eval_int(self) -> Optional[int]:
        lv = self.left.eval_int()
        rv = self.right.eval_int()
        if lv is None or rv is None:
            return None
        return lv + rv

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr

    def eval_int(self) -> Optional[int]:
        lv = self.left.eval_int()
        rv = self.right.eval_int()
        if lv is None or rv is None:
            return None
        return lv * rv

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass(frozen=True)
class FloorDiv(Expr):
    expr: Expr
    divisor: Fraction

    def eval_int(self) -> Optional[int]:
        ev = self.expr.eval_int()
        if ev is None:
            return None
        return math.floor(Fraction(ev, 1) / self.divisor)

    def __str__(self) -> str:
        div = self.divisor
        if div.denominator == 1:
            d = str(div.numerator)
        else:
            d = f"{div.numerator}/{div.denominator}"
        return f"floor({self.expr} / {d})"


def const(n: int) -> Expr:
    return Const(n)


def add(a: Expr, b: Expr) -> Expr:
    if isinstance(a, Const) and a.value == 0:
        return b
    if isinstance(b, Const) and b.value == 0:
        return a
    av = a.eval_int()
    bv = b.eval_int()
    if av is not None and bv is not None:
        return Const(av + bv)
    return Add(a, b)


def mul(a: Expr, k: int) -> Expr:
    if k == 0:
        return Const(0)
    if k == 1:
        return a
    av = a.eval_int()
    if av is not None:
        return Const(av * k)
    if isinstance(a, Mul) and isinstance(a.left, Const):
        return Mul(Const(a.left.value * k), a.right)
    return Mul(Const(k), a)


def floor_div(a: Expr, divisor: Fraction) -> Expr:
    av = a.eval_int()
    if av is not None:
        return Const(math.floor(Fraction(av, 1) / divisor))
    return FloorDiv(a, divisor)


@dataclass
class AddStmt:
    dish: str
    ingredient: str
    count: int
    line: int


@dataclass
class MoveStmt:
    src: str
    dst: str
    line: int


@dataclass
class HeatStmt:
    dish: str
    amount: int
    unit: str  # "min" or "sec"
    line: int


@dataclass
class OutputStmt:
    dish: str
    line: int


@dataclass
class CallStmt:
    name: str
    times: int
    line: int


@dataclass
class IfStmt:
    left: str
    op: str
    right: str
    then_block: List
    else_block: List
    line: int


@dataclass
class Program:
    ingredients: Dict[str, Expr]
    recipes: Dict[str, List]
    main: List


@dataclass
class TraceConfig:
    enabled: bool
    every: int
    max_depth: int
    max_lines: int
    only: Optional[set]
    lines: int = 0


def strip_particle(word: str) -> str:
    w = word.strip()
    if w.endswith("을") or w.endswith("를"):
        return w[:-1]
    return w


def normalize_recipe_name(name: str) -> str:
    n = name.strip()
    if n.endswith("레시피"):
        n = n[: -len("레시피")].rstrip()
    return n


def parse_condition(text: str) -> Optional[Tuple[str, str, str]]:
    # form 1: 만약 A가 B보다 내용물이 ...면:
    m = re.match(r"^만약\s+(.+?)(?:가|이)\s+(.+?)보다\s+내용물이\s+(.+?)면:?\s*$", text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        comp = m.group(3).strip()
        return left, comp, right

    # form 2: 만약 A와 B의 내용물이 ...면:
    m = re.match(r"^만약\s+(.+?)(?:와|과)\s+(.+?)의\s+내용물이\s+(.+?)면:?\s*$", text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        comp = m.group(3).strip()
        return left, comp, right
    return None


def cmp_to_op(comp: str) -> str:
    c = comp.replace(" ", "")
    if "많거나같" in c:
        return ">="
    if "적거나같" in c:
        return "<="
    if "많" in c:
        return ">"
    if "적" in c:
        return "<"
    if "같" in c:
        return "=="
    if "다르" in c:
        return "!="
    return "?"


def parse_program(lines: List[str], unknowns: Optional[List[Tuple[int, str]]] = None) -> Program:
    ingredients: Dict[str, Expr] = {}
    recipes: Dict[str, List] = {}
    main: List = []

    in_table = False
    stack: List[Tuple[str, List]] = []

    def current_block() -> List:
        return stack[-1][1] if stack else main

    for idx, raw in enumerate(lines, start=1):
        s = raw.strip()
        if in_table:
            if s == "":
                continue
            if "=" in s and not s.endswith(":"):
                name, val = [p.strip() for p in s.split("=", 1)]
                if val == "?":
                    ingredients[name] = Var(name)
                else:
                    try:
                        ingredients[name] = Const(int(val))
                    except ValueError:
                        ingredients[name] = Var(name)
                continue
            in_table = False

        if s == "":
            continue

        if s.startswith("칼로리 테이블"):
            in_table = True
            continue

        m = re.match(r"^(.+?)\s*레시피:\s*$", s)
        if m:
            name = m.group(1).strip()
            block: List = []
            recipes[name] = block
            stack.append(("recipe", block))
            continue

        if s == "요리시작:" or s == "요리시작:":
            stack.append(("main", main))
            continue

        cond = parse_condition(s)
        if cond:
            left, comp, right = cond
            op = cmp_to_op(comp)
            then_block: List = []
            else_block: List = []
            stmt = IfStmt(left=left, op=op, right=right, then_block=then_block, else_block=else_block, line=idx)
            current_block().append(stmt)
            stack.append(("if_then", then_block))
            continue

        if s.startswith("그렇지 않으면"):
            if not stack or stack[-1][0] != "if_then":
                raise EvalError(f"else without if at line {idx}")
            stack.pop()
            # find the corresponding IfStmt
            if_stmt = None
            for stmt in reversed(current_block()):
                if isinstance(stmt, IfStmt):
                    if_stmt = stmt
                    break
            if if_stmt is None:
                raise EvalError(f"cannot find if for else at line {idx}")
            stack.append(("if_else", if_stmt.else_block))
            continue

        if s.startswith("끝"):
            if stack:
                stack.pop()
            continue

        m = re.match(r"^(.+?)\s+레시피를\s+만든다\.$", s)
        if m:
            name = normalize_recipe_name(m.group(1))
            current_block().append(CallStmt(name=name, times=1, line=idx))
            continue

        m = re.match(r"^(.+?)\s*(을|를)\s*(\d+)번\s+만든다\.$", s)
        if m:
            name = normalize_recipe_name(strip_particle(m.group(1)))
            times = int(m.group(3))
            current_block().append(CallStmt(name=name, times=times, line=idx))
            continue

        m = re.match(r"^(.+?)\s*(을|를)\s+만든다\.$", s)
        if m:
            name = normalize_recipe_name(strip_particle(m.group(1)))
            current_block().append(CallStmt(name=name, times=1, line=idx))
            continue

        m = re.match(r"^(.+?)에\s+(.+?)(?:\s+(\d+)번)?\s+넣는다\.$", s)
        if m:
            dish = m.group(1).strip()
            ingredient = strip_particle(m.group(2))
            count = int(m.group(3)) if m.group(3) else 1
            current_block().append(AddStmt(dish=dish, ingredient=ingredient, count=count, line=idx))
            continue

        m = re.match(r"^(.+?)의\s+내용물을\s+(.+?)(?:으로|로)\s+옮긴다\.$", s)
        if m:
            src = strip_particle(m.group(1))
            dst = strip_particle(m.group(2))
            current_block().append(MoveStmt(src=src, dst=dst, line=idx))
            continue

        m = re.match(r"^(.+?)(을|를)\s+(\d+)분간\s+가열한다\.$", s)
        if m:
            dish = strip_particle(m.group(1))
            amount = int(m.group(3))
            current_block().append(HeatStmt(dish=dish, amount=amount, unit="min", line=idx))
            continue

        m = re.match(r"^(.+?)(을|를)\s+(\d+)초간\s+가열한다\.$", s)
        if m:
            dish = strip_particle(m.group(1))
            amount = int(m.group(3))
            current_block().append(HeatStmt(dish=dish, amount=amount, unit="sec", line=idx))
            continue

        m = re.match(r"^(.+?)(을|를)\s+식탁\s+위에\s+올려두었다\.$", s)
        if m:
            dish = strip_particle(m.group(1))
            current_block().append(OutputStmt(dish=dish, line=idx))
            continue

        if unknowns is not None:
            unknowns.append((idx, s))
        continue

    return Program(ingredients=ingredients, recipes=recipes, main=main)


def eval_condition(left: Expr, op: str, right: Expr) -> bool:
    lv = left.eval_int()
    rv = right.eval_int()
    if lv is None or rv is None:
        raise UnknownCondition("condition depends on variables")
    if op == ">":
        return lv > rv
    if op == "<":
        return lv < rv
    if op == ">=":
        return lv >= rv
    if op == "<=":
        return lv <= rv
    if op == "==":
        return lv == rv
    if op == "!=":
        return lv != rv
    raise EvalError(f"unknown operator: {op}")


def format_dishes(dishes: Dict[str, Expr]) -> str:
    if not dishes:
        return "<empty>"
    parts = []
    for name in sorted(dishes):
        val = dishes[name].eval_int()
        parts.append(f"{name}={val if val is not None else dishes[name]}")
    return ", ".join(parts)


def safe_filename(text: str) -> str:
    return re.sub(r"[\\/:*?\"<>| ]+", "_", text).strip("_")


def execute(
    program: Program,
    trace: TraceConfig,
    progress_every: int = 0,
    progress_lines: Optional[List[str]] = None,
    snapshot_dir: Optional[str] = None,
) -> List[Tuple[str, Expr]]:
    dishes: Dict[str, Expr] = {}
    outputs: List[Tuple[str, Expr]] = []
    call_stack: List[str] = []
    exec_steps = 0

    def get_dish(name: str) -> Expr:
        return dishes.get(name, Const(0))

    def set_dish(name: str, expr: Expr) -> None:
        dishes[name] = expr

    def trace_state(label: str, call_name: str) -> None:
        if not trace.enabled:
            return
        if trace.only and call_name not in trace.only:
            return
        if trace.max_depth and len(call_stack) > trace.max_depth:
            return
        if trace.max_lines and trace.lines >= trace.max_lines:
            return
        scope = " > ".join(call_stack) if call_stack else "MAIN"
        items = ", ".join(f"{k}={dishes[k]}" for k in sorted(dishes))
        print(f"TRACE {scope}: {label} | {items}")
        trace.lines += 1

    def log_progress(stmt_line: int) -> None:
        nonlocal exec_steps
        exec_steps += 1
        if not progress_every:
            return
        if exec_steps % progress_every != 0:
            return
        msg = f"현재 {exec_steps}행 실행 중 (src {stmt_line}): {format_dishes(dishes)}"
        print(msg)
        if progress_lines is not None:
            progress_lines.append(msg)

    def exec_block(block: List) -> None:
        for stmt in block:
            if isinstance(stmt, AddStmt):
                ing = program.ingredients.get(stmt.ingredient)
                if ing is None:
                    ing = Var(stmt.ingredient)
                expr = add(get_dish(stmt.dish), mul(ing, stmt.count))
                set_dish(stmt.dish, expr)
            elif isinstance(stmt, MoveStmt):
                src_val = get_dish(stmt.src)
                dst_val = get_dish(stmt.dst)
                set_dish(stmt.dst, add(dst_val, src_val))
                set_dish(stmt.src, Const(0))
            elif isinstance(stmt, HeatStmt):
                if stmt.unit == "min":
                    set_dish(stmt.dish, mul(get_dish(stmt.dish), stmt.amount))
                else:
                    if 60 % stmt.amount == 0:
                        divisor = Fraction(60 // stmt.amount, 1)
                    else:
                        divisor = Fraction(60, stmt.amount)
                    set_dish(stmt.dish, floor_div(get_dish(stmt.dish), divisor))
            elif isinstance(stmt, OutputStmt):
                outputs.append((stmt.dish, get_dish(stmt.dish)))
                if snapshot_dir is not None:
                    idx = len(outputs)
                    dish_name = safe_filename(stmt.dish)
                    file_name = f"output_{idx}_{dish_name}_line_{stmt.line}.log"
                    path = os.path.join(snapshot_dir, file_name)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"output[{idx}] {stmt.dish} line={stmt.line}\n")
                        f.write(f"state: {format_dishes(dishes)}\n")
                        if progress_lines:
                            f.write("progress:\n")
                            for line in progress_lines:
                                f.write(line + "\n")
            elif isinstance(stmt, CallStmt):
                if stmt.name not in program.recipes:
                    raise EvalError(f"unknown recipe '{stmt.name}' at line {stmt.line}")
                if stmt.times == 1:
                    call_stack.append(stmt.name)
                    exec_block(program.recipes[stmt.name])
                    call_stack.pop()
                    trace_state(f"after call {stmt.name}", stmt.name)
                else:
                    for i in range(1, stmt.times + 1):
                        call_stack.append(stmt.name)
                        exec_block(program.recipes[stmt.name])
                        call_stack.pop()
                        if i == 1 or i == stmt.times or (trace.every and i % trace.every == 0):
                            trace_state(f"after call {stmt.name} #{i}/{stmt.times}", stmt.name)
            elif isinstance(stmt, IfStmt):
                left = get_dish(stmt.left)
                right = get_dish(stmt.right)
                if eval_condition(left, stmt.op, right):
                    exec_block(stmt.then_block)
                else:
                    exec_block(stmt.else_block)
            else:
                raise EvalError(f"unknown statement at line {getattr(stmt, 'line', '?')}")
            log_progress(getattr(stmt, "line", 0))

    exec_block(program.main)
    return outputs


def apply_overrides(ingredients: Dict[str, Expr], overrides: Dict[str, int]) -> None:
    for k, v in overrides.items():
        ingredients[k] = Const(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cooking code with symbolic variables.")
    parser.add_argument("files", nargs="+", help="input .txt files")
    parser.add_argument("--var", action="append", default=[], help="override constants, e.g. --var 은빛가루=10")
    parser.add_argument("--trace", action="store_true", help="print intermediate states after recipe calls")
    parser.add_argument("--trace-every", type=int, default=0, help="trace every Nth call for repeated recipes")
    parser.add_argument("--trace-max-depth", type=int, default=3, help="max call depth to trace")
    parser.add_argument("--trace-max-lines", type=int, default=2000, help="max trace lines to print")
    parser.add_argument("--trace-only", action="append", default=[], help="trace only these recipe names")
    parser.add_argument("--progress-every", type=int, default=0, help="print state every N executed statements")
    parser.add_argument("--snapshot-dir", default="", help="write progress snapshots on output to this directory")
    parser.add_argument("--strict", action="store_true", help="fail on unparsed lines")
    parser.add_argument("--strict-max-lines", type=int, default=50, help="max unparsed lines to print")
    args = parser.parse_args()

    overrides: Dict[str, int] = {}
    for item in args.var:
        if "=" not in item:
            raise SystemExit(f"invalid --var: {item}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = int(v.strip())

    for path in args.files:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        unknowns: Optional[List[Tuple[int, str]]] = [] if args.strict else None
        program = parse_program(lines, unknowns=unknowns)
        if args.strict and unknowns:
            print(f"UNPARSED lines ({len(unknowns)}):", file=sys.stderr)
            for line_no, text in unknowns[: args.strict_max_lines]:
                print(f"{line_no}: {text}", file=sys.stderr)
            remaining = len(unknowns) - args.strict_max_lines
            if remaining > 0:
                print(f"... {remaining} more", file=sys.stderr)
            raise SystemExit("unparsed lines found")
        apply_overrides(program.ingredients, overrides)

        snapshot_dir = None
        progress_lines = None
        if args.snapshot_dir:
            base_dir = args.snapshot_dir
            if len(args.files) > 1:
                stem = os.path.splitext(os.path.basename(path))[0]
                base_dir = os.path.join(base_dir, stem)
            os.makedirs(base_dir, exist_ok=True)
            snapshot_dir = base_dir
            progress_lines = []

        only = None
        if args.trace_only:
            only = set()
            for item in args.trace_only:
                parts = [p.strip() for p in item.split(",") if p.strip()]
                only.update(parts)
        trace = TraceConfig(
            enabled=args.trace,
            every=args.trace_every,
            max_depth=args.trace_max_depth,
            max_lines=args.trace_max_lines,
            only=only,
        )
        outputs = execute(
            program,
            trace,
            progress_every=args.progress_every,
            progress_lines=progress_lines,
            snapshot_dir=snapshot_dir,
        )

        print(f"== {path} ==")
        for i, (dish, expr) in enumerate(outputs, start=1):
            print(f"output[{i}] {dish} = {expr}")


if __name__ == "__main__":
    main()
