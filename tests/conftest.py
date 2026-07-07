import psycopg
import pytest

from foodsafety_rag import store

ADMIN_URL = "postgresql://postgres:postgres@localhost:5432/postgres"
TEST_URL = "postgresql://postgres:postgres@localhost:5432/foodsafety_test"


@pytest.fixture(scope="session")
def db_conn():
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres not running - start it with: docker compose up -d db")
    if not admin.execute(
        "SELECT 1 FROM pg_database WHERE datname = 'foodsafety_test'"
    ).fetchone():
        admin.execute("CREATE DATABASE foodsafety_test")
    admin.close()

    conn = store.get_conn(TEST_URL)
    store.init_schema(conn)
    yield conn
    conn.close()
