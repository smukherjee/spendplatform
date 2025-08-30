"""Simple database utilities for thread-safe SQLite operations."""
import sqlite3
from contextlib import contextmanager
from typing import Generator
from src.config import config
from src.exceptions.base import DatabaseError
from src.utils.debug import debug_logger


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Get a thread-safe database connection.
    
    Yields:
        A SQLite connection object
        
    Raises:
        DatabaseError: If connection cannot be established
    """
    connection = None
    try:
        debug_logger.debug("Opening database connection", {"path": config.database.path})
        connection = sqlite3.connect(
            config.database.path,
            check_same_thread=False  # Allow connections across threads
        )
        connection.row_factory = sqlite3.Row
        debug_logger.debug("Database connection established successfully")
        yield connection
    except sqlite3.Error as e:
        debug_logger.exception("Database operation failed", e)
        if connection:
            connection.rollback()
        raise DatabaseError(f"Database operation failed: {str(e)}")
    finally:
        if connection:
            connection.close()
