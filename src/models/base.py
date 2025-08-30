"""Base model class for database entities."""
from datetime import datetime
from typing import Any, Dict, Optional
from src.utils.db_simple import get_db_connection
from src.exceptions.base import DatabaseError, NotFoundError


class BaseModel:
    """Base class for all database models."""
    
    table_name: str = ""
    primary_key: str = "id"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize model with attribute values."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def get_by_id(cls, id_value: Any) -> Optional['BaseModel']:
        """Get a record by its primary key.
        
        Args:
            id_value: The primary key value
            
        Returns:
            Model instance if found, None otherwise
            
        Raises:
            DatabaseError: If database operation fails
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {cls.table_name} WHERE {cls.primary_key} = ?",
                (id_value,)
            )
            row = cursor.fetchone()
            return cls(**dict(row)) if row else None

    def save(self) -> None:
        """Save the current instance to the database.
        
        Raises:
            DatabaseError: If save operation fails
        """
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get all instance attributes that are not None
            attrs = {k: v for k, v in self.__dict__.items() 
                    if not k.startswith('_') and v is not None}
            
            if hasattr(self, self.primary_key):
                # Update existing record
                set_clause = ", ".join(f"{k} = ?" for k in attrs.keys())
                values = tuple(attrs.values())
                cursor.execute(
                    f"UPDATE {self.table_name} SET {set_clause} "
                    f"WHERE {self.primary_key} = ?",
                    values + (getattr(self, self.primary_key),)
                )
            else:
                # Insert new record
                columns = ", ".join(attrs.keys())
                placeholders = ", ".join("?" * len(attrs))
                values = tuple(attrs.values())
                cursor.execute(
                    f"INSERT INTO {self.table_name} ({columns}) "
                    f"VALUES ({placeholders})",
                    values
                )
            
            conn.commit()

    def delete(self) -> None:
        """Delete the current instance from the database.
        
        Raises:
            NotFoundError: If record doesn't exist
            DatabaseError: If delete operation fails
        """
        if not hasattr(self, self.primary_key):
            raise NotFoundError(self.table_name, "No primary key")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE {self.primary_key} = ?",
                (getattr(self, self.primary_key),)
            )
            if cursor.rowcount == 0:
                raise NotFoundError(
                    self.table_name,
                    str(getattr(self, self.primary_key))
                )
            conn.commit()

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_')}
