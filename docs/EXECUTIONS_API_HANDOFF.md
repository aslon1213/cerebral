# Cerebral executions API — handoff

You are continuing a build that is partway through a plan. Stages 0 and 1 are
done and tested. Your job is stages 2 onward, in order.

Read this whole file before writing code. The conventions and the gotchas
sections exist because each item in them already cost someone an hour.

---

## 1. What this system is

Cerebral tracks what AI agents do to codebases.

A **task** gets one or more **executions** — a run of an agent (or a human)
against that task. An execution appends an append-only log of **events**: chat
messages, reasoning, tool calls, decisions. Some events carry **code changes**,
which are *pointers into git*, not content. An execution may block on an
**intervention**: an approval or a question that a human must answer.

The point of the whole design: a user opens a file months later and asks "why is
this line here?" The answer comes from joining a code change back to the event
that produced it, and therefore to the agent's reasoning.

**The git model.** Agents never commit to the default branch. They commit under
`refs/cerebral/executions/<execution_id>` — a ref namespace outside
`refs/heads/*`, so the commits are invisible to normal git tooling until the run
finishes. Then the range is squashed into one or more commits on the default
branch, recorded in `execution_repos.landed_commit_shas`.

**Division of labour: git stores content, Postgres stores coordinates and
meaning.** The database holds blob OIDs, commit SHAs, paths and reasoning. It
does not hold file contents. `CodeChange.diff` is an optional rendering cache
only — the real patch is always `git diff <before_blob> <after_blob>`.

**The client** is an observer bot (`libs/observer/`) that watches a Claude Code
session and posts events. It is invoked with `--pwd --api_key --task_id
--project_id --agent_id`. It authenticates with an API key, never a JWT.

---

## 2. What is already done

### Models — complete and tested

| file | holds |
|---|---|
| `app/repo/execution/execution.py` | `Execution`, `ExecutionRepoLink`, `ExecutionError`, `ExecutionStatus`, `ExecutorType`, `OID` |
| `app/repo/execution/history.py` | `ExecutionEvent`, `ExecutionEventPayload`, `ExecutionEventType`, `ActorType`, `allocate_event_seq()` |
| `app/repo/execution/codebase.py` | `CodeChange`, `CodeDiffApplied`, `ChangeType` |
| `app/repo/execution/interrupts.py` | `ExecutionIntervention`, `InterventionKind`, `InterventionStatus` |
| `app/repo/git_repo.py` | `GitRepo`, `GitRepoStore`, `CEREBRAL_REF_NAMESPACE`, `cerebral_ref()` |
| `app/repo/agent.py` | `Agent`, `AgentStore` |
| `app/repo/api_keys.py` | `ApiKey`, `ApiKeyScope`, `ApiKeyStore` |
| `app/repo/types.py` | `PydanticJSONB`, `StrEnumType` |

The schema is heavily constrained on purpose. Check constraints encode a state
machine; composite foreign keys stop rows referring across executions; RESTRICT
protects the audit trail. **Trust the constraints and let them reject bad data —
do not re-implement them in Python.** But do catch `IntegrityError` at the store
boundary and re-raise a domain error, so a client mistake is a 409, not a 500.

### Stage 0 — API-key auth (done)

- `app/core/security.py` — `generate_api_key`, `hash_api_key`, `parse_api_key`, `verify_api_key`
- `app/core/deps.py` — `Principal`, `ApiKeyAuth`, `require_scopes(...)`
- `app/routes/api_keys.py` — `POST/GET/DELETE /api-keys`, `GET /api-keys/verify`

Key format `cbrl_<12-hex-prefix>_<secret>`. Prefix stored in the clear and
uniquely indexed; only SHA-256 of the whole key is stored. SHA-256 rather than
argon2 is deliberate and commented — a 256-bit CSPRNG secret has no dictionary
to attack, and argon2 would tax the ingest path.

`Principal` is the unified caller: `user_id` always, plus `kind`
(`user` | `api_key`), `agent_id`, `key_id`, `scopes`. Ownership checks read
`principal.user_id` regardless of how the caller authenticated.

### Stage 1 — agents and repos (done)

- `app/routes/agents.py` — CRUD + `GET /agents/by-name/{name}`
- `app/routes/git_repos.py` — `POST /repos` (idempotent connect), CRUD,
  `GET /repos/by-name/{name}`, and `GET /repos?local_path=` — the exact-match
  lookup the bot uses to turn `--pwd` into a repo id.

`POST /repos` returns 201 when it created the row and 200 when it already
existed, with `created: bool` in the body. Reconnecting refreshes `local_path`.

### Tests

`tests/` runs against **real Postgres**, never mocks or SQLite — the schema's
value is in constraints that exist nowhere else.

- DSN: `postgresql+asyncpg://postgres:postgres@localhost:55432/cerebral_test`
- The suite creates that database if missing, builds the schema from the models
  with `create_all`, truncates between tests, drops everything at the end.
- Fixtures in `tests/conftest.py`: `engine`, `session`, `seed`, `blob`,
  `client` (anonymous), `user_client` (JWT for the seed user).
- Files: `test_execution_lifecycle.py`, `test_execution_constraints.py`,
  `test_api_keys.py`, `test_agents.py`, `test_git_repos.py`.

Run: `.venv/bin/python -m pytest -q`. Everything passes right now. If a count
drops, you broke something.

### Migrations

**The user runs Alembic themselves.** Do not run `alembic upgrade` or
`alembic revision`. When you change a model, say so explicitly at the end of
your turn so they can generate the migration. Tests use `create_all`, so they
pass regardless — that is exactly why it is easy to forget.

---

## 3. Conventions — follow these exactly

### Module layout

One module per entity holding **model + Create/Update/Response + Store**, like
`app/repo/task.py` and `app/repo/agent.py`. Do not split into `models.py` /
`schemas.py` / `repositories.py`.

New store classes are named `*Store` (`GitRepoStore`, `AgentStore`,
`ApiKeyStore`). Older ones are `*Repo` (`TaskRepo`, `ProjectRepo`, `LabelRepo`)
— leave those alone, but name anything new `*Store`.

### Responses

Every endpoint returns `Response[T]` built with `ok(...)`. Lists return
`Response[Page[T]]`. Failures are raised as `HTTPException` or as a `RepoError`
subclass, never constructed by hand.

```python
@router.get("/{thing_id}", responses=error_responses(404))
async def get_thing(...) -> Response[ThingResponse]:
    thing = await things.get(thing_id, current_user.id)
    if thing is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Thing not found")
    return ok(ThingResponse.model_validate(thing))
```

Declare failure statuses with `error_responses(...)` on the router (shared ones)
and per route (specific ones), so the OpenAPI docs describe errors the way they
actually arrive.

### Domain errors

Three steps, all required:

1. A `RepoError` subclass in `app/repo/base.py`, with a message written for a
   person.
2. A member in `ErrorCode` in `app/core/response.py`.
3. An entry in `REPO_ERRORS` in `app/core/errors.py` mapping it to a status.

The store raises it; the handler in `errors.py` turns it into the envelope. A
route should not catch it.

### Ownership

Path-parameter dependencies: `TaskAccess`, `ProjectAccess`, `LabelAccess`,
`AgentAccess`, `GitRepoAccess` in `app/core/deps.py`. They 404 for a missing row
and 403 for someone else's. Add a matching one for any new path-addressed
entity. Stores also take `owner_id` and filter by it — belt and braces.

### Enums

Always three things together:

```python
class ThingStatus(StrEnum):
    ACTIVE = "active"

# in __table_args__
enum_check("status", ThingStatus, "status")

# on the column
status: ThingStatus = Field(sa_type=StrEnumType(ThingStatus, 16))
```

Never a bare `StrEnum` annotation without `sa_type` — see gotcha 1.

### Constraint names

The metadata naming convention in `app/repo/base.py` prepends `ck_<table>_`.
Pass the **bare** name: `name="blobs_match_type"`, not
`name="ck_code_changes_blobs_match_type"`. Doubling it produces
`ck_x_ck_x_...` and truncates at Postgres's 63-character identifier limit.

### Columns

- Timestamps: `sa_type=DateTime(timezone=True)`. A bare `datetime | None`
  compiles to `TIMESTAMP WITHOUT TIME ZONE`.
- Nullable JSONB: `JSONB(none_as_null=True)`, or `PydanticJSONB` which sets it.
- Git OIDs: the `OID` alias (`String(64)`). Never `CHAR`.
- Append-only tables: `CreatedAtMixin`, not `TimestampMixin` — no `updated_at`
  to tempt anyone into an UPDATE.

### Comments

Explain *why*, not *what*. The existing code comments the non-obvious decision
and the trap avoided. Match that density — do not narrate obvious code, and do
not leave a subtle choice unexplained.

---

## 4. Gotchas that already bit us

1. **SQLModel maps a bare `StrEnum` to a native Postgres enum whose labels are
   the member NAMES.** `ChangeType.CREATED = "created"` becomes a PG enum with
   label `'CREATED'`, so `CHECK (change_type = 'created')` fails at
   `CREATE TABLE`. Always use `StrEnumType`.

2. **`JSONB` stores Python `None` as the JSON scalar `'null'`, not SQL NULL**,
   unless `none_as_null=True`. `col IS NULL` is then false for a column that
   looks empty, silently breaking check constraints and filters.

3. **`CHAR(n)` is blank-padded by Postgres.** A 40-hex SHA-1 in a `CHAR(64)`
   comes back with 24 trailing spaces and fails every Python comparison.

4. **`commit()` expires ORM instances.** Read ids into locals *before*
   committing, or reading an attribute afterwards raises
   `DetachedInstanceError`. The `Seed` fixture holds ids only for this reason.

5. **`bindparams` needs a real `uuid.UUID`**, not the string that arrived in
   JSON. asyncpg binds a str as varchar and Postgres has no `uuid = varchar`
   operator.

6. **asyncpg cannot run two statements in one `execute`.** Split them.

7. **`session.exec()` is typed for SELECTs.** For DML with RETURNING, use
   `connection = await session.connection()` then `connection.execute(...)` —
   it still joins the session's transaction, which is what holds the row lock.
   See `allocate_event_seq`.

8. **This FastAPI version keeps included routes in a nested `_IncludedRouter`**,
   so `app.routes` looks empty. Do not conclude routing is broken; test with the
   httpx client instead.

9. **`ck_api_keys_expires_after_created` means a key cannot be created already
   expired.** To test expiry you must backdate `created_at` as well.

10. **`HTTPBearer` returns 401, not 403**, for a missing Authorization header in
    this FastAPI version.

11. **`user_client` and `client` are separate httpx instances.** A test that
    needs both an authenticated and an anonymous caller gets a genuinely
    anonymous one.

---

## 5. Decisions already made — do not relitigate

- **0.1** The observer bot authenticates with an **API key** in `X-API-Key`.
  Execution write routes require the key, not a JWT. Management routes (issuing
  keys, creating agents) require a JWT, so a leaked key cannot mint
  replacements.
- **0.2** Several interventions may be pending on one execution at once — a real
  agent batches tool approvals. The unique index was dropped. `waiting_*` means
  "at least one open", and clears when the last resolves.
- **0.3** The git repo store is named `GitRepoStore`.
- **0.4** Routes are **flat**, not nested. `POST /executions` carries both
  `task_id` and `project_id` in the body; validate that the task belongs to
  that project and the project to the caller.
- **0.5** **No batch event append for now.** Sequential single appends only.
  Batch can come later.

---

## 6. Your work — stages 2 to 7

Do them in order. Stop after each stage, report what you did, and let the user
review before starting the next.

### Stage 2 — execution lifecycle

**2.1** `ExecutionStore` in `app/repo/execution/execution.py`: `create`
(allocates `attempt`, attaches repo links in the same transaction), `get`,
`list`, `transition`, `add_usage`, `attach_repo`, `set_head`, `land`.

**2.2** `POST /executions` — one call starts a run:

```jsonc
{ "project_id": "…", "task_id": "…",
  "executor_type": "ai_agent",
  "model": "claude-opus-5", "provider": "anthropic",
  "additional_context": { },
  "repos": [ { "repo_id": "…", "base_commit_sha": "…" } ] }
```

`executor_agent_id` comes from the API key's `agent_id`, not the body — a bot
must not be able to claim an agent it was not issued for. Server fills
`ref_name` from `cerebral_ref(execution_id)`. Returns 201.

**2.3** `GET /executions` — `?task_id=&project_id=&status=&agent_id=&repo_id=`,
sortable, `Page[ExecutionResponse]`.

**2.4** `GET /executions/{id}` — with repo links inlined.

**2.5** Transitions, one legal state change each. **No generic `PATCH` with a
writable `status`** — the check constraints are a state machine, and a free-form
patch turns a client mistake into a constraint 500 instead of a 409:

`POST /executions/{id}/start` · `/complete` · `/fail` (requires an
`ExecutionError` body) · `/cancel`

Reject an illegal transition in the store with a domain error → 409, before the
database sees it.

**2.6** `POST /executions/{id}/usage` — incremental token/cost accumulation, no
state change.

**2.7** `DELETE /executions/{id}` — 409 while history exists; `?purge=true`
deletes in dependency order (code_changes → events → repo links → execution).
`test_history_can_still_be_dropped_deliberately` shows the order.

**2.8** New `ErrorCode`: `invalid_transition`, `history_exists`.

**Auth**: writes require `ApiKeyAuth` + `require_scopes(EXECUTIONS_WRITE)`.
Reads should accept either an API key or a JWT — you will need a combined
dependency; add it to `deps.py` next to `ApiKeyAuth`.

### Stage 3 — event ingest (the bot's hot path)

**3.1** `EventStore` in `history.py`: `append` (allocate seq → insert event →
insert code changes → backfill `payload.code_changes`, **one transaction**),
`list_after`, `get`.

**3.2** `POST /executions/{id}/events` — code changes ride along, so an event
and its changes are never half-written:

```jsonc
{ "client_event_id": "bot-run7-0042",
  "event_type": "code_change", "actor_type": "agent",
  "parent_event_id": null, "cerebral_commit_sha": "…",
  "payload": { "reasoning": "jwt verify was missing", "data": { } },
  "code_changes": [ { "repo_id": "…", "path": "app/auth.py",
                      "change_type": "modified",
                      "before_blob": "…", "after_blob": "…",
                      "lines_added": 12, "lines_deleted": 3 } ] }
```

**Idempotency is the critical behaviour here.** `client_event_id` makes the
append safe to retry: **201** when it created the event, **200 with the existing
event** when it was a replay. Implement as `ON CONFLICT (execution_id,
client_event_id) DO NOTHING` then re-select. A bot over an unreliable channel
will replay, and the transcript must not double.

Use `allocate_event_seq()` — it already exists and is tested. Do not compute
`max(seq) + 1`.

**3.3** `GET /executions/{id}/events` — **cursor paginated on `seq`**, not
offset. It is an append-only log; offsets shift under concurrent appends.
`?after_seq=&limit=&event_type=&actor_type=`.

**3.4** `GET /executions/{id}/events/{event_id}` — with code changes expanded.

**3.5** No `PUT`/`PATCH`/`DELETE` on events. A correction is a new event.

### Stage 4 — interventions

**4.1** `InterventionStore` in `interrupts.py`: `open` (intervention +
`intervention_requested` event + execution moves to `waiting_*`, one
transaction), `resolve`, `list_pending`, `list_for_execution`.

**4.2** `POST /executions/{id}/interventions` — the agent asks. Several may be
open at once (decision 0.2).

**4.3** `GET /interventions` — **the inbox**: everything pending across every
execution, oldest first. Backed by `ix_execution_interventions_pending`. JWT
only — this is a human's queue.

**4.4** `GET /executions/{id}/interventions`.

**4.5** Three verbs, not one `respond` with a status field:
`POST /interventions/{id}/approve` · `/reject` · `/answer`.
`ck_execution_interventions_status_matches_kind` already says approvals resolve
approved/rejected and questions resolve answered — separate routes make the
illegal combination unrepresentable rather than a 400.

Each appends `intervention_resolved`, sets `resolved_by_user_id`/`resolved_at`,
and returns the execution to `running` **only when no other intervention is
still pending**. One transaction.

**JWT only, never an API key.** A bot must not answer its own approval requests
— that is why `OBSERVER_BOT_SCOPES` deliberately omits any such scope.

**4.6** New `ErrorCode`: `intervention_already_resolved`,
`intervention_kind_mismatch`.

### Stage 5 — git state and history

**5.1** `POST /executions/{id}/repos` — attach a repo mid-run.
**5.2** `GET /executions/{id}/repos` — ref, base, head, landed state per repo.
**5.3** `POST /executions/{id}/repos/{repo_id}/head` — advance
`head_commit_sha` as the agent commits under `refs/cerebral/*`.
**5.4** `POST /executions/{id}/repos/{repo_id}/land` — record the squash
(`landed_branch`, `landed_commit_shas[]`, `merge_commit_sha`) **and** stamp
`landed_commit_sha` on that repo's code changes. One transaction.
**5.5** `GET /executions/{id}/changes` — everything the run touched, by repo.
**5.6** `GET /repos/{repo_id}/history?path=` — **the endpoint the whole design
exists for.** Every agent change to a file across all executions, with the
reasoning, both blob OIDs and both commit SHAs. The query is already written and
proven in `test_history_of_a_file_carries_the_reasoning` — start from it.
Returns pointers, not content: the client renders the diff from local git.
**5.7** `GET /repos/{repo_id}/history/{change_id}` — one change with full event
context.

### Stage 6 — hardening (only after 2–5 work)

**6.1** Rate limiting on the ingest path.
**6.2** Intervention expiry sweep — a job, not an endpoint.
**6.3** `GET /executions/{id}/events/stream` — SSE tail. Needs `LISTEN/NOTIFY`
or polling and changes how the app deploys. **Ask before building this.**

### Stage 7 — tests

Write tests **with each stage**, not at the end. Extend `tests/`, same real-
Postgres fixtures.

- `tests/test_executions.py` — creation, listing, every legal transition, and
  every illegal one returning 409 rather than a constraint 500.
- `tests/test_events.py` — append, **replay returns 200 with the same event
  id**, cursor pagination across a concurrent append, code changes written in
  the same transaction as their event.
- `tests/test_interventions.py` — ask, inbox, all three resolutions, kind/status
  mismatch, several pending at once, and that an API key **cannot** resolve one.
- `tests/test_history.py` — file history across two executions and two repos.

Test what the constraints reject, not only the happy path — that is what caught
the two real bugs already fixed (`none_as_null` and the enum round-trip).

---

## 7. Working agreement

- Stop after each stage and report. Do not run ahead.
- Never run Alembic. Say when a model changed so the user can migrate.
- Prefer the Edit tool over shell scripts that patch files — the user reviews
  diffs and opaque `python3 - <<EOF` rewrites are hard to check.
- If a design question comes up that stages 2–7 do not answer, ask rather than
  guessing. Several of the decisions above were reversed once already; a wrong
  guess is expensive.
