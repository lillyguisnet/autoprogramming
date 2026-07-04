import inspect
from typing import Annotated, TypeAliasType, get_type_hints, get_args, get_origin
from dataclasses import dataclass


# --- Type-safe signature introspection ---

@dataclass
class Field:
    name: str
    type: type
    description: str | None = None

@dataclass
class Signature:
    name: str
    inputs: list[Field]
    outputs: list[Field]

    def __repr__(self):
        ins = ", ".join(f"{f.name}: {f.type.__name__}" for f in self.inputs)
        outs = ", ".join(f"{f.name}: {f.type.__name__}" for f in self.outputs)
        return f"{self.name}({ins}) -> ({outs})"


def _unwrap(typ):
    """Unwrap TypeAliasType (from `type X = ...` statements) to the underlying type."""
    if isinstance(typ, TypeAliasType):
        return typ.__value__
    return typ

def _resolve_field(typ, fallback_name: str | None = None) -> Field:
    typ = _unwrap(typ)
    if get_origin(typ) is Annotated:
        args = get_args(typ)
        base, label = args[0], args[1]
        desc = args[2] if len(args) > 2 else None
        return Field(name=label, type=base, description=desc)
    return Field(name=fallback_name or "result", type=typ)


def introspect(obj, attr: str) -> Signature:
    """Extract full signature from annotation + lambda params."""
    hints = get_type_hints(type(obj), include_extras=True)
    hint = hints[attr]
    # Access raw lambda from class dict to avoid method binding (eats 1st param as self)
    func = type(obj).__dict__[attr]
    param_names = list(inspect.signature(func).parameters.keys())

    # The annotation IS the output type (or tuple of output types)
    hint = _unwrap(hint)
    if get_origin(hint) is tuple:
        outputs = [_resolve_field(t) for t in get_args(hint)]
    else:
        outputs = [_resolve_field(hint)]

    # Inputs come from lambda param names — type is str by default
    inputs = [Field(name=p, type=str) for p in param_names]

    return Signature(name=attr, inputs=inputs, outputs=outputs)


def introspect_all(obj) -> dict[str, Signature]:
    hints = get_type_hints(type(obj), include_extras=True)
    sigs = {}
    for attr in hints:
        if callable(getattr(obj, attr, None)):
            sigs[attr] = introspect(obj, attr)
    return sigs


# --- Define output types inline, no classes needed ---

type Search = Annotated[str, "Search"]
type Report = Annotated[str, "Report", "Final research report"]


# --- Clean DSPy-like syntax: annotation = output, lambda = inputs ---

class ResearchAgent:
    planner:    Annotated[str, "Plan", "A step-by-step research plan"] = lambda question: ...
    searcher:   Search         = lambda plan: ...
    writer:     Report         = lambda query, question: ...
    summarizer: tuple[planner.Plan, Report] = lambda text: ...


# --- Demo ---

if __name__ == "__main__":
    agent = ResearchAgent()
    sigs = introspect_all(agent)

    for name, sig in sigs.items():
        print(sig)
        for f in sig.inputs:
            print(f"  IN  {f.name}: {f.type.__name__}" + (f"  # {f.description}" if f.description else ""))
        for f in sig.outputs:
            print(f"  OUT {f.name}: {f.type.__name__}" + (f"  # {f.description}" if f.description else ""))
        print()
