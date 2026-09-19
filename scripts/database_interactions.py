import sqlite3


def database_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys = ON;')
    return conn


def database_create(db_path: str, sql_path: str) -> None:
    # Create database with the tables from the provided file
    with open(sql_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    with database_connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()


def database_write(db_path: str, sql: str, params) -> None:
    # Runs INSERT/UPDATE/DELETE, returning last updated row id

    with database_connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(sql, params)
        conn.commit()


def database_read(db_path: str, sql: str, params: tuple = ()) -> list:
    # Runs SELECT on the database, returning all the corrosponding rows.

    with database_connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()
