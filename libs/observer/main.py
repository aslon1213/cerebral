import argparse
import asyncio
import re
import os
import platform
from pathlib import Path

import time
import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


logger = structlog.get_logger()

parser = argparse.ArgumentParser()

parser.add_argument("--pwd")
parser.add_argument("--session_id", required=False)
parser.add_argument("--api_key")
parser.add_argument("--task_id")
parser.add_argument("--project_id")
parser.add_argument("--agent_id")


class FileUpdateHandler(FileSystemEventHandler):
    def __init__(self, file_to_watch):  # pyright: ignore[reportUnknownParameterType]
        self.file_to_watch = file_to_watch

    def on_modified(self, event):
        # Watchdog monitors directories; filter for your specific file
        logger.debug(
            "fs.modified", src_path=event.src_path, watching=self.file_to_watch
        )
        if event.src_path.endswith(self.file_to_watch):
            logger.info("session.file_changed", src_path=event.src_path)
            # Insert your custom code here (e.g., re-read the file)


def sluggify(path_str: str):
    """
    Converts an absolute directory path into a Claude Code compatible project slug.
    Replaces every individual non-alphanumeric character with a single dash.
    """
    # Standardize to an absolute path first
    absolute_path = os.path.abspath(path_str)

    # Replace any non-alphanumeric character with a dash
    slug = re.sub(r"[^a-zA-Z0-9]", "-", absolute_path)
    logger.debug("path.sluggify", path=path_str, absolute_path=absolute_path, slug=slug)
    return slug


def get_claude_code_os_path(working_directory_slug):  # pyright: ignore[reportUnknownParameterType]
    current_os = platform.system()
    if current_os == "windows":
        raise NotImplementedError()
        return ""
    elif current_os == "Darwin":
        path = os.path.join(
            os.path.expanduser("~/.claude/projects"), working_directory_slug
        )
        logger.debug(
            "path.claude_projects",
            system=current_os,
            slug=working_directory_slug,
            path=path,
        )
        return path
    elif current_os == "Linux":
        raise NotImplementedError()
        return ""
    else:
        raise ValueError("Unsupported System; Only Windows, MacOS/Linux are supported.")


def get_latest_changed_session(working_directory_slug):  # pyright: ignore[reportUnknownParameterType]
    folder_path = get_claude_code_os_path(working_directory_slug=working_directory_slug)
    d = Path(folder_path)
    files = [p for p in d.iterdir() if p.is_file()]
    logger.debug("session.scan", dir=str(d), count=len(files))
    latest = max(files, key=lambda p: p.stat().st_mtime)
    logger.debug("session.latest", path=str(latest), mtime=latest.stat().st_mtime)
    return latest.stem


async def main():
    args = parser.parse_args()
    pwd = args.pwd
    pwd_slugged = sluggify(pwd)
    session_id = args.session_id
    logger.info("startup", pwd=pwd, slug=pwd_slugged, session_id=session_id)

    if not session_id:
        # get latest changed file
        logger.info("session.resolving_latest")
        session_id = get_latest_changed_session(pwd_slugged)

    watch_file = f"{session_id}.jsonl"
    watch_path = get_claude_code_os_path(pwd_slugged)

    events_handlers = FileUpdateHandler(watch_file)
    observer = Observer()
    observer.schedule(event_handler=events_handlers, path=watch_path)
    observer.start()
    logger.info(
        "observer.started",
        session_id=session_id,
        watch_file=watch_file,
        watch_path=watch_path,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("observer.stopping")
        observer.stop()
    observer.join()
    logger.info("observer.stopped")


if __name__ == "__main__":
    asyncio.run(main())
