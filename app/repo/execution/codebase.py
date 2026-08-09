import uuid
from datetime import datetime
from enum import StrEnum

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from pydantic import Field as PydanticField
from sqlmodel import DateTime, Field, String

from app.repo.base import CreatedAtMixin, enum_check
from app.repo.types import PydanticJSONB, StrEnumType

from .execution import OID, GitSha


class ChangeType(StrEnum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class CodeDiffApplied(BaseModel):
    """One replaced region. line_start/line_end are 1-based, inclusive,
    in NEW-file coordinates (after this change was applied).

    Optional cache for rendering. Git is the source of truth: the exact patch
    is always recoverable with ``git diff <before_blob> <after_blob>``, so this
    is only worth populating when the UI needs to show a hunk without shelling
    out. Leave it NULL for large changes rather than storing megabytes of text
    that already exist in the object store.
    """

    old: str
    new: str
    line_start: int
    line_end: int


class CodeChange(CreatedAtMixin, table=True):
    """One file touched by one event, as a pointer into git.

    The division of labour: git stores content (blobs, commits, the full patch),
    this table stores the coordinates and the semantics -- which execution,
    which event, and therefore which reasoning, produced this edit. That is what
    lets a user open a file and ask "why is this line here?"; the answer comes
    from joining back to execution_events, not from anything stored here.
    """

    __tablename__ = "code_changes"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        # The lookup behind "show me this file's agent history". Scoped by repo:
        # two repos both have a src/main.py, and an unscoped index puts them in
        # the same range.
        sa.Index(
            "ix_code_changes_repo_path_created",
            "repo_id",
            "path",
            sa.text("created_at DESC"),
        ),
        # A change belongs to a repo the execution actually works in. The
        # composite FK gets that for free instead of leaving repo_id free to
        # disagree with execution_repos.
        sa.ForeignKeyConstraint(
            ["execution_id", "repo_id"],
            ["execution_repos.execution_id", "execution_repos.repo_id"],
            name="fk_code_changes_execution_repo",
            ondelete="RESTRICT",
        ),
        # seq is optional and ordinal within an event: an agent that batches
        # parallel tool calls into a single event still gets a stable order,
        # while an agent that does not can leave it NULL. Postgres treats NULLs
        # as distinct, so several unordered changes per event are allowed.
        sa.UniqueConstraint("event_id", "seq", name="uq_code_changes_event_seq"),
        enum_check("change_type", ChangeType, "change_type"),
        sa.CheckConstraint(
            "(change_type = 'created'  AND before_blob IS NULL     AND after_blob IS NOT NULL)"
            " OR (change_type = 'deleted'  AND before_blob IS NOT NULL AND after_blob IS NULL)"
            " OR (change_type IN ('modified','renamed') AND before_blob IS NOT NULL AND after_blob IS NOT NULL)",
            name="blobs_match_type",
        ),
        sa.CheckConstraint(
            "(change_type = 'renamed') = (previous_path IS NOT NULL)",
            name="rename_has_previous_path",
        ),
        sa.CheckConstraint(
            "lines_added >= 0 AND lines_deleted >= 0",
            name="line_counts_non_negative",
        ),
        sa.CheckConstraint(
            "reverts_change_id IS NULL OR reverts_change_id <> id",
            name="revert_not_self",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # Both RESTRICT: this is the record of what an agent did to a codebase, and
    # it must not disappear as a side effect of deleting the run it came from.
    execution_id: uuid.UUID = Field(
        foreign_key="executions.id", index=True, ondelete="RESTRICT"
    )
    event_id: uuid.UUID = Field(
        foreign_key="execution_events.id", index=True, ondelete="RESTRICT"
    )
    repo_id: uuid.UUID = Field(foreign_key="git_repos.id", index=True)

    seq: int | None = Field(default=None)
    change_type: ChangeType = Field(sa_type=StrEnumType(ChangeType, 16))  # pyright: ignore[reportArgumentType]
    path: str = Field(sa_type=String(1024))  # pyright: ignore[reportArgumentType]
    previous_path: str | None = Field(default=None, sa_type=String(1024))  # pyright: ignore[reportArgumentType]
    language: str | None = Field(default=None, sa_type=String(64))  # pyright: ignore[reportArgumentType]

    before_blob: str | None = Field(default=None, sa_type=OID)  # pyright: ignore[reportArgumentType]
    after_blob: str | None = Field(default=None, sa_type=OID)  # pyright: ignore[reportArgumentType]
    # Where this change ended up in the default namespace, once the execution's
    # cerebral range was merged. NULL until then, and the same value for every
    # change of one repo in one run, since the range becomes a single commit.
    #
    # The commit that actually produced this change is the cerebral-side one,
    # and it is not repeated here: it lives on execution_events.cerebral_commit_sha
    # and is reachable through event_id. The two are the same hash from the same
    # namespace, so storing it twice would only create a way for them to differ.
    landed_commit_sha: str | None = Field(default=None, sa_type=OID)  # pyright: ignore[reportArgumentType]

    lines_added: int = Field(default=0)
    lines_deleted: int = Field(default=0)
    diff: list[CodeDiffApplied] | None = Field(
        default=None,
        sa_column=sa.Column(PydanticJSONB(CodeDiffApplied, many=True), nullable=True),
    )

    applied_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))  # pyright: ignore[reportArgumentType]
    # A revert is its own change, not an edit of the row being reverted --
    # mutating history would defeat the point of keeping it. This points at the
    # change being undone.
    reverts_change_id: uuid.UUID | None = Field(
        default=None, foreign_key="code_changes.id", ondelete="RESTRICT"
    )


# --------------------------------------------------------------------------- #
# Request / response models
# --------------------------------------------------------------------------- #
class CodeChangeCreate(BaseModel):
    """One file an event touched, as it arrives with that event.

    Never posted on its own: a change is only meaningful next to the reasoning
    that produced it, so it rides along with its event and the two are written
    in one transaction.

    No ``seq`` field. The ordinal is positional -- the order of this list is the
    order the changes happened in -- so letting a client send its own would only
    create a way for the two to disagree.
    """

    repo_id: uuid.UUID
    change_type: ChangeType
    path: str = PydanticField(min_length=1, max_length=1024)
    previous_path: str | None = PydanticField(default=None, max_length=1024)
    language: str | None = PydanticField(default=None, max_length=64)
    before_blob: GitSha | None = None
    after_blob: GitSha | None = None
    lines_added: int = PydanticField(default=0, ge=0)
    lines_deleted: int = PydanticField(default=0, ge=0)
    # Optional rendering cache. Leave it out for a large change rather than
    # posting megabytes of text that git already holds.
    diff: list[CodeDiffApplied] | None = None
    applied_at: datetime | None = None
    reverts_change_id: uuid.UUID | None = None


class CodeChangeResponse(BaseModel):
    """A change as a set of git coordinates, never as content.

    ``before_blob`` and ``after_blob`` are what a client renders the diff from,
    against its own checkout: ``git diff <before_blob> <after_blob>``.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    execution_id: uuid.UUID
    event_id: uuid.UUID
    repo_id: uuid.UUID
    seq: int | None
    change_type: ChangeType
    path: str
    previous_path: str | None
    language: str | None
    before_blob: str | None
    after_blob: str | None
    landed_commit_sha: str | None
    lines_added: int
    lines_deleted: int
    diff: list[CodeDiffApplied] | None
    applied_at: datetime | None
    reverts_change_id: uuid.UUID | None
    created_at: datetime
