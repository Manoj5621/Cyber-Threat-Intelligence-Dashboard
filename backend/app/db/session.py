import ssl
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from ..config import settings

DATABASE_URL = (
    f"mysql+pymysql://{settings.mysql_user}:"
    f"{settings.mysql_password}@"
    f"{settings.mysql_host}:"
    f"{settings.mysql_port}/"
    f"{settings.mysql_database}"
)

# Engine (2.x compatible)
if settings.backend_env == "production":
    connect_args = {
        "ssl": {
            "ca": settings.MYSQL_SSL_CA,
            "check_hostname": False,
            "verify_mode": ssl.CERT_NONE
        }
    }
else:
    connect_args = {}

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    future=True,
    connect_args=connect_args
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False
)

# Base model (2.x style)
class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
