"""Custom SQLAlchemy column types shared by the repository models."""

from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import JSONB


class StrEnumType(sa.types.TypeDecorator[Any]):
    """Store a ``StrEnum`` as ``VARCHAR``, read it back as the enum member.

    Storage stays a plain string -- native Postgres enums make adding a value an
    ``ALTER TYPE`` that cannot run in a transaction with other DDL, and these
    vocabularies grow. But a bare ``String`` column hands back a ``str`` on
    read, so ``execution.status is ExecutionStatus.FAILED`` is quietly always
    False while ``==`` works. Round-tripping through the enum removes that trap.

    Pair with :func:`app.repo.base.enum_check` so the database rejects values
    the enum does not define.
    """

    impl = sa.String
    cache_ok = True

    def __init__(self, enum_cls: type[StrEnum], length: int, **kwargs: Any) -> None:
        self.enum_cls = enum_cls
        longest = max(len(member.value) for member in enum_cls)
        if longest > length:
            raise ValueError(
                f"{enum_cls.__name__} has a {longest}-char value but the column "
                f"is String({length})"
            )
        super().__init__(length=length, **kwargs)

    def process_bind_param(self, value: Any, dialect: sa.Dialect) -> Any:
        if value is None:
            return None
        return self.enum_cls(value).value

    def process_result_value(self, value: Any, dialect: sa.Dialect) -> Any:
        if value is None:
            return None
        return self.enum_cls(value)


class PydanticJSONB(sa.types.TypeDecorator[Any]):
    """Store a Pydantic model in a ``JSONB`` column.

    A bare ``JSONB`` column cannot hold a model instance: SQLAlchemy serialises
    with ``json.dumps``, which raises ``TypeError`` on a ``BaseModel``. This
    decorator dumps on the way in and validates on the way out, so the attribute
    is a typed model in Python and plain JSON in Postgres (still queryable with
    the ``->>`` / ``@>`` operators).

    Pass ``many=True`` for a JSON array of models.

    Note: the value is *replaced*, never mutated in place. Editing a nested
    field will not mark the attribute dirty, so assign a whole new model to
    persist a change. The tables using this are append-only, so that is the
    intended usage anyway.
    """

    impl = JSONB
    cache_ok = True

    def __init__(
        self, model: type[BaseModel], *, many: bool = False, **kwargs: Any
    ) -> None:
        self.model = model
        self.many = many
        # none_as_null: without it SQLAlchemy stores Python None as the JSON
        # scalar 'null' rather than SQL NULL, and `col IS NULL` is then false
        # for a column that looks empty -- which silently breaks every check
        # constraint and filter written against it.
        kwargs.setdefault("none_as_null", True)
        super().__init__(**kwargs)

    @staticmethod
    def _dump(value: Any) -> Any:
        # mode="json" so UUIDs/datetimes/Decimals become JSON scalars.
        return value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    def process_bind_param(self, value: Any, dialect: sa.Dialect) -> Any:
        if value is None:
            return None
        if self.many:
            return [self._dump(item) for item in value]
        return self._dump(value)

    def process_result_value(self, value: Any, dialect: sa.Dialect) -> Any:
        if value is None:
            return None
        if self.many:
            return [self.model.model_validate(item) for item in value]
        return self.model.model_validate(value)
