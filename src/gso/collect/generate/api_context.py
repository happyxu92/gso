"""Parent-revision API metadata collection and prompt formatting."""

from __future__ import annotations

import json

from gso.collect.generate.helpers import count_tokens

MAX_PARENT_API_CONTEXT_TOKENS = 15000


def build_parent_api_preflight_test(
    apis: list[str], repo_name: str, result_prefix: str
) -> str:
    """Build a test that resolves APIs and inspects the installed parent revision."""
    script = r'''import ast
import importlib
import importlib.metadata
import inspect
import json
import re
import subprocess
import timeit
import typing
from pathlib import Path

TARGET_APIS = __TARGET_APIS__
REPO_NAME = __REPO_NAME__
RESULT_PREFIX = __RESULT_PREFIX__
REPO_PATH = Path("/workspace") / REPO_NAME
ALIASES = {
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib.pyplot",
    "tf": "tensorflow",
    "torch": "torch",
}
IGNORED_SOURCE_PARTS = {
    ".git", ".venv", "venv", "build", "dist", "node_modules", "site-packages"
}
MAX_SOURCE_CHARS = 12000
MAX_EXAMPLE_CHARS = 1500
MAX_TYPE_SOURCE_CHARS = 3000
MAX_EXAMPLES = 5
MAX_TYPES = 6
_SOURCE_RECORDS = None


def _normalize_distribution_name(name):
    return name.lower().replace("-", "_").replace(".", "_")


def _getattr_chain(value, attributes):
    for attribute in attributes:
        value = getattr(value, attribute)
    return value


def _repository_roots():
    normalized_repo = _normalize_distribution_name(REPO_NAME)
    roots = {normalized_repo}
    try:
        distributions = importlib.metadata.packages_distributions()
    except Exception:
        distributions = {}
    for package, names in distributions.items():
        if any(_normalize_distribution_name(name) == normalized_repo for name in names):
            roots.add(package)

    if REPO_PATH.is_dir():
        for package_parent in (REPO_PATH, REPO_PATH / "src"):
            if not package_parent.is_dir():
                continue
            for child in package_parent.iterdir():
                if child.is_dir() and (child / "__init__.py").is_file():
                    roots.add(child.name)
    return sorted(roots)


def _module_defines_api(source_path, api_root):
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == api_root:
                return True
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == api_root for target in node.targets):
                return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == api_root:
                return True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[-1]) == api_root for alias in node.names):
                return True
    return False


def _candidate_modules(root_module, api_root):
    candidates = []
    for raw_package_path in getattr(root_module, "__path__", []):
        package_path = Path(raw_package_path)
        for source_path in package_path.rglob("*.py"):
            if source_path.name == "__main__.py":
                continue
            if not _module_defines_api(source_path, api_root):
                continue
            relative = source_path.relative_to(package_path)
            suffix_parts = list(relative.with_suffix("").parts)
            if suffix_parts[-1] == "__init__":
                suffix_parts.pop()
            module_name = ".".join([root_module.__name__, *suffix_parts])
            candidates.append(module_name)
    return sorted(set(candidates), key=lambda name: (name.count("."), name))


def _resolve_api(api):
    parts = api.split(".")
    errors = []

    alias_module = ALIASES.get(parts[0])
    if alias_module is not None:
        try:
            return _getattr_chain(importlib.import_module(alias_module), parts[1:])
        except (Exception, SystemExit) as error:
            errors.append(f"{alias_module}: {type(error).__name__}: {error}")

    for split_index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            module = importlib.import_module(module_name)
            return _getattr_chain(module, parts[split_index:])
        except (Exception, SystemExit) as error:
            errors.append(f"{module_name}: {type(error).__name__}: {error}")

    roots = _repository_roots()
    for root_name in roots:
        try:
            root_module = importlib.import_module(root_name)
            return _getattr_chain(root_module, parts)
        except (Exception, SystemExit) as error:
            errors.append(f"{root_name}: {type(error).__name__}: {error}")
            try:
                root_module = importlib.import_module(root_name)
            except (Exception, SystemExit):
                continue

        for module_name in _candidate_modules(root_module, parts[0]):
            try:
                module = importlib.import_module(module_name)
                return _getattr_chain(module, parts)
            except (Exception, SystemExit):
                continue

    detail = "; ".join(errors[-8:])
    raise ImportError(
        f"could not resolve {api!r} from repository distribution roots {roots}"
        + (f"; recent errors: {detail}" if detail else "")
    )


def _truncate(text, limit):
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...(truncated)...", True


def _display_path(path):
    if not path:
        return None
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(REPO_PATH.resolve()))
    except (OSError, ValueError):
        return str(candidate)


def _source_records():
    global _SOURCE_RECORDS
    if _SOURCE_RECORDS is not None:
        return _SOURCE_RECORDS
    records = []
    if not REPO_PATH.is_dir():
        _SOURCE_RECORDS = records
        return records
    paths = list(REPO_PATH.rglob("*.pyi")) + list(REPO_PATH.rglob("*.py"))
    for path in paths:
        try:
            relative = path.relative_to(REPO_PATH)
        except ValueError:
            continue
        if any(part in IGNORED_SOURCE_PARTS for part in relative.parts):
            continue
        try:
            if path.stat().st_size > 1000000:
                continue
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        records.append((path, relative, source, tree))
    _SOURCE_RECORDS = records
    return records


def _node_source(source, node, limit):
    segment = ast.get_source_segment(source, node)
    if segment is None:
        lines = source.splitlines()
        segment = "\n".join(lines[node.lineno - 1 : getattr(node, "end_lineno", node.lineno)])
    return _truncate(segment, limit)


def _ast_signature(node):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    try:
        signature = f"({ast.unparse(node.args)})"
        if node.returns is not None:
            signature += f" -> {ast.unparse(node.returns)}"
        return signature
    except Exception:
        return None


def _static_definition(name, limit):
    parts = [part for part in name.split(".") if part]
    target = parts[-1]
    owner = parts[-2] if len(parts) > 1 else None
    module_hints = [parts[:-1], parts[:-2]]
    candidates = []
    for path, relative, source, tree in _source_records():
        relative_module = list(relative.with_suffix("").parts)
        if relative_module and relative_module[-1] == "__init__":
            relative_module.pop()
        module_rank = 2
        for hint in module_hints:
            if hint and len(relative_module) >= len(hint) and relative_module[-len(hint) :] == hint:
                module_rank = 0
                break
        path_rank = 0 if path.suffix == ".pyi" else 1
        if any(part.lower().startswith("test") for part in relative.parts):
            path_rank += 3
        for node in tree.body:
            matched = None
            if owner and isinstance(node, ast.ClassDef) and node.name == owner:
                matched = next(
                    (
                        child
                        for child in node.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and child.name == target
                    ),
                    None,
                )
            if matched is None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == target:
                    matched = node
            if matched is None and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(item, ast.Name) and item.id == target for item in targets):
                    matched = node
            if matched is not None:
                candidates.append(
                    (module_rank, path_rank, len(relative.parts), str(relative), source, matched)
                )
    if not candidates:
        return None
    _, _, _, relative, source, node = min(candidates, key=lambda item: item[:4])
    definition, truncated = _node_source(source, node, limit)
    return {
        "source_path": relative,
        "source_line": node.lineno,
        "source": definition,
        "source_truncated": truncated,
        "static_signature": _ast_signature(node),
    }


def _runtime_source(value):
    try:
        target = inspect.unwrap(value)
    except Exception:
        target = value
    try:
        lines, line = inspect.getsourcelines(target)
        source, truncated = _truncate("".join(lines), MAX_SOURCE_CHARS)
        return {
            "source_path": _display_path(inspect.getsourcefile(target) or inspect.getfile(target)),
            "source_line": line,
            "source": source,
            "source_truncated": truncated,
        }
    except (OSError, TypeError):
        return None


def _dotted_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _import_aliases(tree):
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _normalize_call_name(name, aliases):
    parts = name.split(".")
    replacement = aliases.get(parts[0])
    return ".".join([replacement, *parts[1:]]) if replacement else name


def _documentation_call_examples(api, resolved, limit):
    if limit <= 0 or not REPO_PATH.is_dir():
        return []
    names = sorted({api, resolved, api.split(".")[-1]}, key=len, reverse=True)
    patterns = [
        (name, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\("))
        for name in names
        if name
    ]
    matches = []
    for suffix in ("*.md", "*.rst", "*.txt"):
        for path in REPO_PATH.rglob(suffix):
            try:
                relative = path.relative_to(REPO_PATH)
            except ValueError:
                continue
            if any(part in IGNORED_SOURCE_PARTS for part in relative.parts):
                continue
            try:
                if path.stat().st_size > 1000000:
                    continue
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                matched_name = next(
                    (name for name, pattern in patterns if pattern.search(line)),
                    None,
                )
                if matched_name is None:
                    continue
                start = max(1, line_number - 3)
                end = min(len(lines), line_number + 3)
                snippet, truncated = _truncate(
                    "\n".join(lines[start - 1 : end]), MAX_EXAMPLE_CHARS
                )
                matches.append(
                    {
                        "source_path": str(relative),
                        "source_line": line_number,
                        "call": matched_name,
                        "match": "documentation",
                        "source": snippet,
                        "source_truncated": truncated,
                    }
                )
                if len(matches) == limit:
                    return matches
    return matches


def _call_examples(api, resolved):
    target = api.split(".")[-1]
    matches = []
    for path, relative, source, tree in _source_records():
        if path.suffix != ".py":
            continue
        aliases = _import_aliases(tree)
        lines = source.splitlines()
        path_parts = {part.lower() for part in relative.parts}
        path_rank = 0 if any(part.startswith("test") for part in path_parts) else 1
        if "examples" in path_parts or "example" in path_parts:
            path_rank = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            raw_name = _dotted_name(node.func)
            if raw_name is None:
                continue
            normalized = _normalize_call_name(raw_name, aliases)
            if normalized in {api, resolved}:
                match_rank = 0
                match_kind = "qualified"
            elif normalized.endswith("." + api) or resolved.endswith("." + normalized):
                match_rank = 1
                match_kind = "suffix"
            elif raw_name.split(".")[-1] == target:
                match_rank = 2
                match_kind = "method-name"
            else:
                continue
            start = max(1, node.lineno - 3)
            end = min(len(lines), getattr(node, "end_lineno", node.lineno) + 3)
            snippet, truncated = _truncate("\n".join(lines[start - 1 : end]), MAX_EXAMPLE_CHARS)
            matches.append(
                (
                    match_rank,
                    path_rank,
                    str(relative),
                    node.lineno,
                    {
                        "source_path": str(relative),
                        "source_line": node.lineno,
                        "call": raw_name,
                        "match": match_kind,
                        "source": snippet,
                        "source_truncated": truncated,
                    },
                )
            )
    matches.sort(key=lambda item: item[:4])
    selected = []
    seen = set()
    for _, _, path, line, example in matches:
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        selected.append(example)
        if len(selected) == MAX_EXAMPLES:
            break
    selected.extend(
        _documentation_call_examples(api, resolved, MAX_EXAMPLES - len(selected))
    )
    return selected


def _annotation_text(annotation):
    if annotation is inspect.Signature.empty:
        return None
    try:
        return inspect.formatannotation(annotation)
    except Exception:
        return repr(annotation)


def _collect_annotation_types(annotation, found):
    if annotation is inspect.Signature.empty:
        return
    if isinstance(annotation, str):
        for name in re.findall(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\b", annotation):
            if name not in {"None", "str", "int", "float", "bool", "bytes", "list", "dict", "tuple", "set"}:
                found.setdefault(name, None)
        return
    origin = typing.get_origin(annotation)
    if origin is not None:
        for argument in typing.get_args(annotation):
            _collect_annotation_types(argument, found)
        return
    if inspect.isclass(annotation):
        module = getattr(annotation, "__module__", "")
        name = f"{module}.{annotation.__qualname__}" if module else annotation.__qualname__
        if module not in {"builtins", "typing", "types"}:
            found.setdefault(name, annotation)


def _related_types(signature):
    found = {}
    if signature is not None:
        for parameter in signature.parameters.values():
            _collect_annotation_types(parameter.annotation, found)
        _collect_annotation_types(signature.return_annotation, found)
    definitions = []
    for name, value in found.items():
        if len(definitions) == MAX_TYPES:
            break
        definition = _runtime_source(value) if value is not None else None
        if definition is not None:
            source, truncated = _truncate(definition["source"], MAX_TYPE_SOURCE_CHARS)
            definition = {**definition, "source": source, "source_truncated": truncated}
        else:
            definition = _static_definition(name, MAX_TYPE_SOURCE_CHARS)
        if definition is not None:
            definitions.append({"name": name, **definition})
    return definitions


def _parent_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_PATH, capture_output=True,
            text=True, check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _api_context(api, value):
    resolved = (
        f"{getattr(value, '__module__', '')}."
        f"{getattr(value, '__qualname__', getattr(value, '__name__', ''))}"
    ).strip(".")
    signature_object = None
    signature = None
    signature_error = None
    try:
        signature_object = inspect.signature(value)
        signature = str(signature_object)
    except (TypeError, ValueError) as error:
        text_signature = getattr(value, "__text_signature__", None)
        if text_signature:
            signature = text_signature.strip()
        else:
            signature_error = f"{type(error).__name__}: {error}"

    source = _runtime_source(value)
    static = None
    if source is None or signature is None:
        static = _static_definition(resolved or api, MAX_SOURCE_CHARS)
        if static is None and resolved != api:
            static = _static_definition(api, MAX_SOURCE_CHARS)
    if source is None and static is not None:
        source = {key: value for key, value in static.items() if key != "static_signature"}
    if signature is None and static is not None:
        signature = static.get("static_signature")

    annotations = {}
    if signature_object is not None:
        annotations = {
            name: _annotation_text(parameter.annotation)
            for name, parameter in signature_object.parameters.items()
            if parameter.annotation is not inspect.Signature.empty
        }
        if signature_object.return_annotation is not inspect.Signature.empty:
            annotations["return"] = _annotation_text(signature_object.return_annotation)

    return {
        "parent_commit": _parent_commit(),
        "resolved": resolved,
        "signature": signature,
        "signature_error": signature_error,
        **(source or {"source_path": None, "source_line": None, "source": None, "source_truncated": False}),
        "annotations": annotations,
        "call_examples": _call_examples(api, resolved),
        "related_types": _related_types(signature_object),
    }


def setup():
    return None


def experiment():
    results = {}
    for api in TARGET_APIS:
        try:
            value = _resolve_api(api)
            results[api] = {"ok": True, **_api_context(api, value)}
        except (Exception, SystemExit) as error:
            message = f"{type(error).__name__}: {error}"
            results[api] = {"ok": False, "error": message[-500:]}
    print(RESULT_PREFIX + json.dumps(results, separators=(",", ":")), flush=True)
    failures = [api for api, result in results.items() if not result["ok"]]
    if failures:
        raise RuntimeError("API reference preflight failed: " + ", ".join(failures))
    return results


def store_result(result, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(result, file)


def load_result(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def check_equivalence(reference, current):
    assert reference == current


def run_test(eqcheck=False, reference=False, prefix=""):
    setup()
    execution_time, result = timeit.timeit(experiment, number=1)
    result_path = f"{prefix}_api_preflight.json"
    if reference:
        store_result(result, result_path)
    elif eqcheck:
        check_equivalence(load_result(result_path), result)
    return execution_time
'''
    return (
        script.replace("__TARGET_APIS__", json.dumps(apis, ensure_ascii=False))
        .replace("__REPO_NAME__", json.dumps(repo_name))
        .replace("__RESULT_PREFIX__", json.dumps(result_prefix))
    )


def format_parent_api_context(context: dict | None) -> str:
    """Render bounded, provenance-rich parent metadata for the generation prompt."""
    if not context:
        return ""
    rendered = json.dumps(context, indent=2, ensure_ascii=False)
    if count_tokens(rendered) > MAX_PARENT_API_CONTEXT_TOKENS:
        # Probe fields are independently truncated; this is a final safety bound for
        # unusually verbose signatures or paths.
        rendered = rendered[: MAX_PARENT_API_CONTEXT_TOKENS * 4]
        rendered += "\n...(parent API context truncated)..."
    return (
        "\n\n## Exact Parent-Revision API Context\n"
        "The following data was collected from the installed parent commit. "
        "Use these signatures and repository examples instead of guessing.\n"
        f"```json\n{rendered}\n```\n"
    )
