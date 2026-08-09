"""Importing this package registers every table on the SQLModel metadata.

The models reference each other by name (``Relationship("Project")``), so they
all have to be imported before SQLAlchemy can configure the mappers.
"""

from . import (  # noqa: F401
    agent,
    api_keys,
    execution,
    git_repo,
    labels,
    project,
    task,
    tokens,
    users,
)
