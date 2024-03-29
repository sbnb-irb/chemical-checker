"""Basic connection and query fuctions for PostgreSQL."""
from .psql import get_connection, qstring, qstring_cur, query