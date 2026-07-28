"""Importing this package registers every table on the SQLModel metadata.

The models reference each other by name (``Relationship("Project")``), so they
all have to be imported before SQLAlchemy can configure the mappers.
"""

from . import labels, project, task, tokens, users  # noqa: F401
