"""The ML layer: pure Python, NO FastAPI imports.

Keeping ML code independent of the web framework means you can:
  - test it without HTTP,
  - reuse it in a CLI, a notebook, or a worker,
  - move it into its own pip package later ("an ML library").
"""
