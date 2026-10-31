"""Pydantic schemas: the shape of data going IN (requests) and OUT (responses).

FastAPI uses them to: validate input (bad input -> automatic 422 error),
convert JSON <-> Python objects, filter responses, and generate the /docs page.
"""
