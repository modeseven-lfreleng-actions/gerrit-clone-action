# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""What the operator decided to filter, for a whole clone tree.

A repository's own record (:mod:`gerrit_clone.content_policy`) says which
filters *rewrote* it -- an effect, withdrawn when a filter changed
nothing.  It cannot say what was decided for the tree: a project cloned
later, or a file that only appears upstream later, has no record at all.

This is that decision.  It lives beside the projects, in
``<tree>/.gerrit-clone/filter-policy.json``, so it outlives any one
repository: one deleted and cloned again, by hand or by
``mirror --overwrite``, is filtered from it as before.  It only ever
grows.  Each scope -- every project, a ``--git-filter`` project pattern,
or one project exactly -- holds a :class:`~gerrit_clone.content_policy.\
FilterPolicy`; a project's share is the union of every scope covering it.

Like the repository record, it never holds a token: only the SHA-256
digest of each, to check tokens given again against.  A file that
cannot be read, or that a newer release wrote, is an error, never an
empty decision.

ADR 0001 records the design: ``docs/adr/0001-content-filter-intent.md``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gerrit_clone.content_policy import FilterPolicy
from gerrit_clone.models import match_project_pattern

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

INTENT_DIR = ".gerrit-clone"
INTENT_FILE = "filter-policy.json"
SCHEMA = 1

#: Scope of every project in the tree.
ALL_PROJECTS = ("projects", "*")


class IntentError(RuntimeError):
    """The tree's filter intent could not be read or written.

    Nothing may be cloned, refreshed or mirrored: either way, what the
    tree must be filtered under is unknown.
    """


def intent_path(root: Path) -> Path:
    return root / INTENT_DIR / INTENT_FILE


def find_root(start: Path) -> Path | None:
    """The nearest of *start* and its parents holding an intent file.

    Anything at that path counts, a directory or a broken link included:
    passed over, it would let a run in a subtree choose a weaker root,
    where reading it fails closed.
    """
    here = start.resolve()
    for candidate in (here, *here.parents):
        found = intent_path(candidate)
        if found.exists() or found.is_symlink():
            return candidate
    return None


@dataclass(frozen=True)
class FilterIntent:
    """Filter policy per scope.

    A scope is ``("projects", pattern)``, matched as ``--git-filter``
    and ``--include-projects`` patterns are, or ``("project", name)``,
    matched exactly.
    """

    scopes: Mapping[tuple[str, str], FilterPolicy] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return all(policy.empty for policy in self.scopes.values())

    def for_project(self, name: str) -> FilterPolicy:
        """What was decided for project *name*, from every scope covering it."""
        policy = FilterPolicy()
        for (kind, value), scoped in self.scopes.items():
            if (kind == "project" and value == name) or (
                kind == "projects" and match_project_pattern(name, value)
            ):
                policy = policy.union(scoped)
        return policy

    def union(self, other: FilterIntent) -> FilterIntent:
        scopes = dict(self.scopes)
        for scope, policy in other.scopes.items():
            scopes[scope] = scopes.get(scope, FilterPolicy()).union(policy)
        return FilterIntent(scopes)

    @classmethod
    def of_options(
        cls,
        remove_patterns: Iterable[str] | None,
        git_filter_projects: Mapping[str, Iterable[str]] | None,
        redact_secrets: bool,
    ) -> FilterIntent:
        """The decision a run's ``--remove-files``, ``--git-filter`` and
        ``--redact-secrets`` options make."""
        scopes: dict[tuple[str, str], FilterPolicy] = {}
        everywhere = FilterPolicy.of(list(remove_patterns or ()), [], redact_secrets)
        if not everywhere.empty:
            scopes[ALL_PROJECTS] = everywhere
        for pattern, tokens in (git_filter_projects or {}).items():
            scoped = FilterPolicy.of(None, list(tokens), False)
            if not scoped.empty:
                scopes[("projects", pattern)] = scoped
        return cls(scopes)


def load_intent(root: Path) -> FilterIntent:
    """The intent recorded for the tree at *root*; empty if none is.

    Raises:
        IntentError: If the file exists but cannot be read or understood.
    """
    path = intent_path(root)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if path.is_symlink():
            # A link to nothing is a broken intent, not an absent one.
            raise IntentError(
                f"The filter intent {path} is a link to nothing"
            ) from None
        return FilterIntent()
    except (OSError, UnicodeError) as exc:
        raise IntentError(f"Could not read the filter intent {path}: {exc}") from exc
    try:
        return _decode(json.loads(text))
    except (ValueError, TypeError, KeyError) as exc:
        raise IntentError(
            f"Could not understand the filter intent {path}: {exc}"
        ) from exc


def save_intent(root: Path, intent: FilterIntent) -> None:
    """Write *intent* for the tree at *root*, all of it or none of it.

    Raises:
        IntentError: If it could not be written.
    """
    path = intent_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{INTENT_FILE}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(_encode(intent), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    except OSError as exc:
        raise IntentError(f"Could not write the filter intent {path}: {exc}") from exc


def _encode(intent: FilterIntent) -> dict[str, Any]:
    entries = []
    for (kind, value), policy in sorted(intent.scopes.items()):
        if policy.empty:
            continue
        entry: dict[str, Any] = {kind: value}
        if policy.remove_patterns:
            entry["remove"] = sorted(policy.remove_patterns)
        if policy.token_digests:
            entry["token_sha256"] = sorted(policy.token_digests)
        if policy.redact_secrets:
            entry["redact_secrets"] = True
        entries.append(entry)
    return {"schema": SCHEMA, "entries": entries}


def _decode(data: Any) -> FilterIntent:
    if not isinstance(data, dict):
        raise TypeError("not a JSON object")
    schema = data.get("schema")
    # bool is an int, and 1.0 == 1: only the JSON integer itself will do.
    if type(schema) is not int or schema != SCHEMA:
        raise ValueError(
            f"schema {schema!r} is not {SCHEMA}; written by another release"
        )
    entries = data["entries"]
    if not isinstance(entries, list):
        raise TypeError("entries is not a list")
    scopes: dict[tuple[str, str], FilterPolicy] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"entry is not an object: {entry!r}")
        kinds = [kind for kind in ("projects", "project") if kind in entry]
        if len(kinds) != 1 or not isinstance(entry[kinds[0]], str):
            raise ValueError(f"entry without exactly one scope: {entry!r}")
        redact = entry.get("redact_secrets", False)
        if not isinstance(redact, bool):
            raise TypeError(f"redact_secrets is not true or false: {redact!r}")
        policy = FilterPolicy(
            frozenset(_strings(entry, "remove")),
            frozenset(_strings(entry, "token_sha256")),
            redact,
        )
        scope = (kinds[0], entry[kinds[0]])
        scopes[scope] = scopes.get(scope, FilterPolicy()).union(policy)
    return FilterIntent(scopes)


def _strings(entry: Mapping[str, Any], key: str) -> list[str]:
    values = entry.get(key, [])
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise TypeError(f"{key} is not a list of strings")
    return values
