# Cerebral

Cerebral keeps a record of what AI agents do to your codebase.

When an agent writes code for you, the work survives and the reasoning
disappears. You get a diff and a commit message. Six months later someone opens
the file, finds a line nobody understands, and there is no one to ask.

Cerebral keeps the reasoning, and keeps it attached to the code it produced.

## What it does

**Records the run.** Every agent run is an execution against a task. As it
works, it logs what it did and why: messages, reasoning, tool calls, decisions.
The log is append-only, so the record is what happened, not what someone tidied
up afterwards.

**Connects reasoning to code.** Each change an agent makes is linked to the
moment in the log that produced it. Ask why a line is there and you get the
answer — the agent's reasoning at the moment it wrote that line, plus the task it
was working on and the run it was part of.

**Keeps agent work out of the way until it is ready.** Agents commit to their
own area of the repository, invisible to normal git commands. Nothing touches
your branches until the run is done and the work is landed deliberately. A run
that goes wrong leaves no mess to clean up.

**Lets agents ask.** A run can stop and wait for a person — to approve something
risky, to answer a question, to review the result. It stays parked until someone
responds, then picks up where it left off.

**Tracks the cost.** Tokens and spend per run, so you can see what a task
actually cost.

## What it is useful for

- **Understanding code you did not write.** The history of a file, with the
  reasoning behind each change, not just the diff.
- **Reviewing agent work.** See what an agent did and why before it reaches your
  branch, rather than after.
- **Keeping agents on a leash.** Approve the risky steps, answer the questions,
  leave the rest alone.
- **Knowing what agents cost.** Per task, per run, per agent.
- **An audit trail.** Which agent changed what, when, on whose instruction.

## How you use it

Cerebral is an HTTP API. People sign in and read; agents authenticate with API
keys scoped to what they are allowed to do — writing to the log is a different
permission from approving work, and only a person can approve.

An agent reports to it as it runs. `libs/observer/` is a client that watches a
Claude Code session and posts events as they happen, so an agent does not have to
be written against Cerebral to be recorded by it.

## Status

Under active development. The API covers runs, their logs, code history,
approvals, and agent and repository management.

Requires Python 3.13 and Postgres.
