<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 The Linux Foundation
-->

# ADR 0001: A clone tree records its content-filter intent

- **Status:** Accepted
- **Date:** 2026-10-02
- **Issue:** [#307](https://github.com/lfreleng-actions/gerrit-clone-action/issues/307)

## Context

Content filtering (`--remove-files`, `--git-filter`, `--redact-secrets`)
removes files and secrets from cloned history, usually before `mirror`
pushes it to GitHub. Until this decision, each run filtered with its own
options only. Leaving an option out on a later run undid the filtering:

- A refresh fetches upstream history over the filtered history.
- `mirror --overwrite` deletes each project directory, clones it again
  and pushes it with `git push --mirror`, replacing the filtered GitHub
  repository with unfiltered history. A project directory deleted by
  hand meets the same fate.
- A project created upstream after the original clone, or a file that
  appears upstream later, gets no filtering at all.

PR #305 added a per-repository record, in the repository's git config,
of the filters that **rewrote** it. That record answers whether a
refresh is safe, but it records an *effect*: the tool withdraws a filter
that changed nothing, so the record cannot say what the operator
*decided*. It also disappears with the repository.

## Decision

Two layers, each authoritative for one question.

<!-- markdownlint-disable MD013 -->

| Layer                                                        | Holds                                     | Answers                                 | Lifetime                           |
| ------------------------------------------------------------ | ----------------------------------------- | --------------------------------------- | ---------------------------------- |
| Tree-level intent, `<tree>/.gerrit-clone/filter-policy.json` | Every filter any run applied, by scope    | What the operator decided for this tree | Persistent; only grows             |
| Per-repository record (PR #305, git config)                  | The filters that rewrote this repository  | Whether a refresh would be safe         | Lives and dies with the repository |

<!-- markdownlint-enable MD013 -->

### The intent file

- **Scopes.** Each entry applies to every project (`"projects": "*"`),
  to a `--git-filter` project pattern (matched as `--git-filter` and
  `--include-projects` match), or to one project exactly
  (`"project": "com/parent"`). A project's policy is the union of every
  entry covering it.
- **Only grows.** A run adds its options to the file, and nothing
  removes anything. Dropping a filter means deleting the file *and*
  re-cloning the repositories: a deliberate act, never an omission.
- **No secrets.** Tokens appear only as SHA-256 digests, as in the
  per-repository record.
- **Fail closed.** A file the tool cannot read, cannot parse, or that
  carries an unknown `schema` stops the run before it touches any
  repository. The tool never reads it as "nothing decided".
- **Atomic.** The tool writes a temporary file, flushes it, then renames
  it over the old one.
- **Location.** The tool searches upwards from the output path, as git
  finds `.git`, and creates the file at the output path otherwise.
  Discovery already skips hidden directories, so `.gerrit-clone/` never
  passes for a project.

### Each run

At the start of a run, before it clones, refreshes or deletes anything:

1. Read the intent.
2. Fold in every existing repository's own record, scoped to that
   project. This migrates trees that PR #305's release filtered, and it
   lets `mirror --overwrite` delete a repository without losing what it
   held.
3. Add the run's options.
4. Write the result, unless the run is a dry run, or a refresh of a path
   that does not exist. A run writes under the tree's lock,
   `.gerrit-clone/filter-policy.lock`, held with `flock` (`msvcrt.locking`
   on Windows): it takes the lock, reads the intent again, adds its
   additions and writes. Two runs on one tree, such as a `refresh` and a
   `mirror`, never lose each other's additions. A run waits up
   to two minutes for another to release the lock, then stops with an
   error. A run with nothing to add, and a dry run, never take it.

The run then filters every repository with the union of its options and
the intent for that project, through one shared step
(`content_stage.filter_repository`) that `clone`, `refresh` and `mirror`
all use. A refresh stages every mirror the run filters, including ones
the filters have not rewritten yet: it fetches into a copy, filters the
copy, and publishes only if both succeed, so a failed filter never
leaves new unfiltered content in the mirror. A working copy
(`--no-mirror`) has no bare copy to publish atomically, so one the
filters never rewrote pulls first and gets filtered after; a failed
filter there exits non-zero and leaves the checkout for the operator,
as before this decision.

### Tokens

`--remove-files` and `--redact-secrets` re-apply from the intent alone.
A `--git-filter` token cannot: the intent holds its digest, not the
token. The tool **refuses** a project whose intent includes tokens the
run does not supply, comparing digests: a refresh skips it before
fetching, and the `clone` and `mirror` filter stages count it as failed,
which makes `mirror` abort the batch and push nothing. The operator passes the
tokens again, as for the original clone.

Alternatives rejected: storing tokens, which defeats the purpose; and
storing a reference to where tokens live, such as a secret name, which
couples the tool to one secret store and still cannot verify the value.

### Repositories filtered by v2.2.4 and earlier

Unchanged from PR #305: the tool recognises them by their traces and
refuses them, with re-clone as the remedy. Those releases never recorded
their filters, so neither the intent nor any check of current history
can show that a later run re-applies them. An "adopt" path fails for the
same reason: it could check that history now lacks the paths an operator
names, but not that those cover what those releases removed.

For the same reason, `mirror --overwrite` takes one out of the run
entirely, along with any directory holding one and any repository whose
record it cannot read. Cloned again, it would go to GitHub unfiltered,
over the filtered copy; and even as it stands, `--recreate` would
replace its GitHub repository with it. The tool neither deletes, clones,
filters nor pushes it, leaves its GitHub repository alone, and reports
it as skipped with the reason, until the operator deletes it by hand and
clones it again with the filters it needs; that run then records them.

## Consequences

- A later run without filter options filters as the tree decided,
  including new projects and content that arrives later. Filtering what
  the tool already filtered changes nothing, but costs a filter run for
  each repository in each run.
- `mirror --overwrite` and hand-deleted repositories no longer roll a
  GitHub mirror back to unfiltered history.
- The GitHub Action usually clones into a fresh workspace, so its
  workflow inputs remain its policy record; the file mainly serves
  long-lived trees.

### Deferred

Tracked in
[#309](https://github.com/lfreleng-actions/gerrit-clone-action/issues/309):

- **Per-run history.** The file records the accumulated decision, not a
  log of the runs that made it, nor the refs before and after each
  rewrite.
- **Manifests.** The clone, refresh and mirror manifests do not yet
  summarise the tree's intent; the file itself is the auditable record.
- **Staged working copies.** A working copy pulls before its filters
  run, as described above. Staging it would mean filtering a copy and
  then resetting the checkout to the result.
