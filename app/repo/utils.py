from datetime import datetime, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    print(utcnow())
