# cerebral-lib

Internal shared library for the Cerebral project.

Distribution name: `cerebral-lib` — import name: `cerebral`.

```python
from cerebral import __version__
```

This package is a member of the root uv workspace. It is consumed by the app
via a workspace source (`cerebral-lib = { workspace = true }`), so any change
here is picked up without reinstalling.
