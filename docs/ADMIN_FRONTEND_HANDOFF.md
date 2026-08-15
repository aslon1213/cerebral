# Cerebral admin UI — handoff

You are building the web UI for Cerebral. The API is finished and tested; none
of it needs changing to do this work (with one exception, in §2 — read that
first).

Read this whole file before writing code. §1 draws the line around your scope
and it is the thing most likely to be got wrong.

---

## 1. What you are building, and what you are not

Cerebral records what AI agents do to codebases. Two very different clients talk
to the same API:

- **The observer bot** (`libs/observer/`) — runs alongside an agent and *writes*:
  it opens a run, appends the transcript, records code changes, asks for
  approval. Authenticates with an **API key**.
- **A person** — *reads* that record afterwards, answers what the agents are
  blocked on, and manages the things runs are made of. Authenticates with a
  **session token**.

**You are building the person's side only.** Every write endpoint on the ingest
path is the bot's job and is already implemented. Do not build a UI for it.

Concretely, **do not build**:

| Endpoint | Why not |
|---|---|
| `POST /executions` | A run is started by the agent, not from a form |
| `POST /executions/{id}/start\|complete\|fail\|cancel` | The state machine is driven by the runner |
| `POST /executions/{id}/usage` | Reported by the runner |
| `POST /executions/{id}/events` | The transcript is written by the bot |
| `POST /executions/{id}/interventions` | The *agent* asks; the person answers |
| `POST /executions/{id}/repos`, `.../head`, `.../land` | Git state is recorded by the runner |
| `GET /api-keys/verify` | A bot checking its own key at startup |

These are API-key-only at the server, so a session token gets **401** anyway.
That is deliberate — every line of a transcript has to be attributable to a
credential issued for an agent. If a screen seems to need one of these, you have
misread the scope; ask.

**Judgement call to confirm:** I have included **API key management** (issue,
list, revoke) in your scope. Issuing a key is something a person does in an
admin UI — it is *about* bots without being *for* them. If you would rather that
lived elsewhere, say so and drop §6.3.

Executions are **read-only** to you, with one exception: a person may delete a
run (`DELETE /executions/{id}`), which is a cleanup action and belongs here.

---

## 2. Before you start: one API fix is needed

**CORS blocks every `PATCH` from a browser.** `app/main.py` declares:

```python
allow_methods=["GET", "POST", "PUT", "DELETE"]
```

`PATCH` is missing, so the browser preflight is rejected before your request is
ever sent. Verified:

```
POST   preflight -> 200 OK
PATCH  preflight -> 400 'Disallowed CORS method'
```

This breaks **every update endpoint**: agents, repos, tasks, projects, labels.
Add `"PATCH"` to that list — a one-word change — or you will spend an afternoon
debugging your own fetch wrapper. Flag it to the API owner rather than working
around it.

While you are there: `allow_origins=["*"]` is fine for local development and
must be narrowed before this is deployed anywhere real. Not your call, but worth
raising.

---

## 3. Conventions the whole API follows

**Base URL.** Everything is under `/api/v1`. The dev server runs in localhost:8000

**OpenAPI.** Served at `/openapi.json`, with Swagger at `/docs`. Generate your
client from it rather than hand-writing types — every response model below is in
there with exact field types, and a generated client will not drift.

**Every response is the same envelope**, success or failure:

```jsonc
{
  "ok": true,
  "data": { ... },       // present on success
  "error": null,         // present on failure
  "request_id": "..."    // trace id; show it in error UI
}
```

```jsonc
{
  "ok": false,
  "data": null,
  "error": {
    "code": "invalid_transition",  // branch on this, never on the message
    "message": "An execution that is succeeded cannot become running",
    "details": null                // shape depends on the code
  },
  "request_id": "..."
}
```

`error.code` is a stable contract; `error.message` is written for a reader and
may be reworded at any time. **Show the message, branch on the code.** The codes
you will actually meet are in §7.

**Two kinds of pagination**, and using the wrong one is a real bug:

- *Offset* — `Page<T>`: `{items, total, limit, offset}`. Used by every list
  except the transcript. You get a `total`, so you can render page numbers.
- *Cursor* — `EventPage`: `{items, limit, next_after_seq, has_more}`. Used by
  the event transcript **only**. There is no `total`. Pass the `next_after_seq`
  you were given back as `after_seq`; start at `0`.

  The transcript is appended to while you are reading it. An offset shifts under
  every event that arrives, so page two would skip or repeat whatever landed in
  between. This is also how you tail a live run: ask again with the cursor you
  already hold, and an **empty page means you are caught up**.

**Auth.** `Authorization: Bearer <access_token>`. `POST /auth/login` returns
`{access_token, refresh_token, token_type}`. On `401`, call `POST /auth/refresh`
with the refresh token once and retry; if that also fails, send the user to
login. `POST /auth/logout` revokes the refresh token — call it, do not just drop
the tokens client-side.

Never put an API key in the browser. Keys are shown once at creation for the
user to copy, and are for machines.

**Ownership is enforced server-side.** Everything is scoped to the signed-in
user; another user's anything is `403` or `404`. You never send a user id.

---

## 4. The screens

Ordered by how much they matter. Build in this order.

### 4.1 The inbox — the one screen with urgency

`GET /interventions` is everything waiting on the signed-in user, across every
run, **oldest first** — the agent blocked longest is losing the most time. This
is the home screen and it should carry a badge with the pending count.

Each item is one of three kinds, and the kind decides the control:

| `kind` | What it is | Actions |
|---|---|---|
| `approval` | Agent wants permission before doing something | Approve / Reject |
| `qa_review` | Agent wants its work reviewed | Approve / Reject |
| `input_required` | Agent asked a question | Answer (required) |

- `POST /interventions/{id}/approve` — body `{response?, reasoning?}`
- `POST /interventions/{id}/reject` — body `{response?, reasoning?}`
- `POST /interventions/{id}/answer` — body `{response, reasoning?}` — **`response`
  is required**; answering a question with nothing is the one thing that makes no
  sense.

`request` and `response` are free-form JSON objects — the agent decides the
shape. Render them readably, but do not assume a schema.

Approving a question, or answering an approval, is `409
intervention_kind_mismatch`. Resolving one twice is `409
intervention_already_resolved` — likely two tabs, so refresh rather than showing
a hard error.

Resolving one usually unblocks its run, but not always: several may be open at
once and the run stays parked until the last is dealt with. **Re-read the
execution afterwards rather than assuming it went back to `running`.**

### 4.2 Runs

`GET /executions` — filter by `task_id`, `project_id`, `status`, `agent_id`,
`repo_id`; sort with `sort_by` (`created_at`, `updated_at`, `started_at`,
`finished_at`, `attempt`, `status`) and `order` (`asc`, `desc`); newest first by
default.

A run's `status` is the main thing on screen. It has seven values, and they
group into three states a person cares about:

- **Live** — `pending`, `running`
- **Blocked, needs a human** — `waiting_approval`, `waiting_input` (link
  straight to the intervention)
- **Over** — `succeeded`, `failed`, `cancelled`

`attempt` is the retry number for that task — show it, "attempt 3" is meaningful.
A `failed` run carries a structured `error` with `retryable`, so you can say
whether retrying is worth it without parsing prose.

`GET /executions/{id}` additionally inlines `repos` (the git state). The list
endpoint deliberately does not — a project's worth of runs would fan out into a
query per row. **So: the detail view has git state, the list view does not.**

`DELETE /executions/{id}` refuses with `409 history_exists` if the run recorded
anything. Retry with `?purge=true` to drop its events and code changes too.
Confirm that properly — it destroys an audit record and cannot be undone.

### 4.3 The transcript

`GET /executions/{id}/events` — cursor-paginated (§3), oldest first, optionally
filtered by `event_type` and `actor_type`.

This is a conversation, so render it as one. `actor_type` (`agent`, `user`,
`system`) decides the side; `event_type` decides the treatment — a `reasoning`
event should not look like a `chat_message`, and a `tool_call` should be
collapsible. `payload.reasoning` is the prose; `payload.data` is untyped and
its shape depends on the type.

`parent_event_id` links a `tool_result` back to its `tool_call` — nest them.

`GET /executions/{id}/events/{event_id}` returns the same event with its
`code_changes` expanded. `payload.code_changes` holds the same ids in order,
but the expanded list is the authoritative one.

Live runs: poll with the cursor. `execution.last_event_seq` is the high-water
mark, so you can tell you are behind without fetching anything. There is no
websocket or SSE endpoint — polling is the intended mechanism today.

### 4.4 Code, and the question the product exists to answer

`GET /repos/{repo_id}/history?path=app/auth.py` — every agent change to that
file, across every run, newest first, **each one next to the reasoning that
produced it**. Omit `path` for the whole repo.

This is the payoff. Someone opens a file, finds a line nobody understands, and
this screen tells them why it is there. Give it room: the change on one side,
the agent's reasoning on the other, and a link to the full run.

An entry is `{change, event, attempt, executor_agent_id}` — enough to say "the
nightly bot, on its second attempt" without another request.

`GET /repos/{repo_id}/history/{change_id}` gives one change with its event and
its whole execution.

`GET /executions/{id}/changes` is the same data from the other end: everything
one run touched, grouped by repo.

**The API returns git coordinates, never file content.** A change carries
`before_blob` / `after_blob` (blob ids) and `path`, not the text. `diff` is an
optional cache and is often `null` — do not build a viewer that depends on it.
Render from `lines_added` / `lines_deleted` / `change_type` / `path`, and if you
need the actual patch, that is `git diff <before_blob> <after_blob>` against a
checkout the browser does not have. **Do not design a screen that needs file
contents without raising it first** — that is an API change, not a UI one.

Two commits appear and they are not the same thing:

- `event.cerebral_commit_sha` — the commit that *produced* the change, in the
  agent's private namespace.
- `change.landed_commit_sha` — where the run's work ended up on the default
  branch. `null` until the run lands, and the *same value for every change of
  one repo in one run*.

Agents commit under `refs/cerebral/executions/<id>`, outside `refs/heads/*`,
invisible to normal git tooling until the run lands. A change with
`landed_commit_sha: null` has not reached the default branch — worth showing.

`GET /executions/{id}/repos` is the git state per repo: `base_commit_sha`,
`head_commit_sha`, and once landed, `landed_branch`, `landed_commit_shas`,
`merge_commit_sha`, `landed_at`. It returns a plain list, **not** a `Page`.

### 4.5 The things runs are made of

Ordinary CRUD, all `Page<T>`, all owner-scoped: **projects** (with labels, and
`GET /projects/{id}/tasks`), **tasks** (with labels), **labels**, **agents**,
**repos**. Field lists are in the OpenAPI schema — do not hand-write them.

Two worth knowing: agents and repos have a `by-name` lookup
(`GET /agents/by-name/{name}`), and deleting either fails `409 resource_in_use`
if a run references it, because a run is an audit record.

---

## 5. What good looks like

The product's whole claim is that **the reasoning survives next to the code**.
A UI that shows runs as rows of status badges and hides the transcript three
clicks deep has thrown that away. From any change, the reasoning behind it
should be one click; from any run, the transcript should be the main view rather
than a tab.

The second claim is that **a blocked agent is costing you time**. Pending
interventions should be impossible to miss and answerable in one click from the
inbox, without opening the run.

---

## 6. Endpoint reference — your surface only

All paths prefixed `/api/v1`. All take `Authorization: Bearer <token>`.

### 6.1 Auth
```
POST   /auth/register              {name, password} -> UserResponse
POST   /auth/login                 {name, password} -> {access_token, refresh_token, token_type}
POST   /auth/refresh               {refresh_token}  -> new token pair
POST   /auth/logout                {refresh_token}  -> null
GET    /auth/me                                     -> UserResponse
```

### 6.2 Interventions — the inbox
```
GET    /interventions                        ?limit&offset      -> Page<Intervention>   # pending, oldest first
GET    /interventions/{id}                                      -> Intervention
POST   /interventions/{id}/approve           {response?, reasoning?}
POST   /interventions/{id}/reject            {response?, reasoning?}
POST   /interventions/{id}/answer            {response, reasoning?}
GET    /executions/{id}/interventions        ?status&limit&offset -> Page<Intervention>
```

### 6.3 API keys (see the judgement call in §1)
```
POST   /api-keys      {name, agent_id?, scopes?, expires_at?} -> {key, api_key}   # `key` shown ONCE
GET    /api-keys      ?include_revoked&limit&offset           -> Page<ApiKey>
GET    /api-keys/{id}                                         -> ApiKey
DELETE /api-keys/{id}                                         -> revoke
```
Scopes: `executions:read`, `executions:write`, `repos:read`, `repos:write`. The
default is what an observer bot needs. `key` is never recoverable — the UI must
make "copy this now" unmissable.

### 6.4 Runs (read + delete)
```
GET    /executions             ?task_id&project_id&status&agent_id&repo_id&sort_by&order&limit&offset
GET    /executions/{id}                                  -> Execution + inlined repos
DELETE /executions/{id}        ?purge=true               -> 409 history_exists without purge
```

### 6.5 Transcript
```
GET    /executions/{id}/events            ?after_seq&limit&event_type&actor_type -> EventPage (cursor!)
GET    /executions/{id}/events/{event_id}                                        -> event + code_changes
```

### 6.6 Code
```
GET    /executions/{id}/repos                                  -> list (not a Page)
GET    /executions/{id}/changes    ?repo_id&limit&offset       -> Page<CodeChange>
GET    /repos/{id}/history         ?path&limit&offset          -> Page<HistoryEntry>
GET    /repos/{id}/history/{change_id}                         -> {change, event, execution}
```

### 6.7 Everything else
```
/projects  /tasks  /labels  /agents  /repos      # CRUD, Page<T>, see OpenAPI
```

---

## 7. Vocabulary

```
ExecutionStatus   pending running waiting_approval waiting_input
                  succeeded failed cancelled
EventType         chat_message reasoning decision tool_call tool_result
                  code_change status_change intervention_requested
                  intervention_resolved memory_loaded memory_saved
ActorType         agent user system
InterventionKind  approval qa_review input_required
InterventionStatus pending approved rejected answered expired
ChangeType        created modified deleted renamed
```

Error codes you will meet: `validation_error` (422), `unauthorized`,
`forbidden`, `not_found`, `conflict`, `history_exists`,
`intervention_already_resolved`, `intervention_kind_mismatch`,
`resource_in_use`, `duplicate_agent_name`, `duplicate_repo_name`,
`internal_error`.

---

## 8. Gotchas

1. **`PATCH` is CORS-blocked** until §2 is fixed. Every update fails.
2. **Two paginations.** The transcript is cursored on `seq`; everything else is
   offset. Do not write one generic paginator and use it for both.
3. **`GET /executions` has no repo data.** Only the detail view inlines it.
4. **`GET /executions/{id}/repos` returns a bare list**, not a `Page`.
5. **`diff` is usually `null`.** It is a cache, not the source of truth.
6. **Resolving an intervention may not unblock the run** — several can be open.
   Re-read, do not assume.
7. **`status` on `GET /executions` is a query alias.** Send `?status=running`.
8. **A 404 may mean "not yours".** Ownership failures are deliberately
   indistinguishable from missing. Do not tell the user the thing exists.
9. **`request_id` is on every response.** Put it in your error toast; it is how
   anything gets traced later.
10. **Timestamps are UTC ISO-8601.** Format in the user's zone at the edge.

---

## 9. Open questions — ask, do not guess

- **Framework and styling are not decided.** There is no frontend in this repo
  yet. Propose a stack before you scaffold.
- **API key management in this UI?** See §1.
- **Live updates** are polling-only today. If you want push, say so — it is an
  API change (SSE was deliberately deferred) and not something to work around
  with a 1-second poll.
- **Rendering real diffs** needs file content the API does not serve, by design.
  Raise it rather than designing around it.

If a question comes up that this file does not answer, ask. Several decisions in
the API were reversed once already; a wrong guess is expensive.
