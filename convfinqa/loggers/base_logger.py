import os


class SingletonMeta(type):
    """A metaclass for creating singleton classes."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Create a new instance of the class if it does not exist."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseLogger(metaclass=SingletonMeta):
    """A base class for creating loggers."""

    def __init__(
        self,
        name: str,
        level: int = None,
        log_file_path: str = None,
        log_file_backup_count: int = None,
        log_interval: int = None,
        log_interval_unit: str = None,
    ) -> None:
        self._name = name
        self._level = level or int(os.getenv("LOG_LEVEL", 10))
        self._log_file_path = log_file_path or "./"
        self._filename = os.path.join(self._log_file_path, f"{self._name}.log")
        self._log_file_backup_count = log_file_backup_count or int(
            os.getenv("LOG_FILE_BACKUP_COUNT", 14)
        )
        self._log_interval = log_interval or int(os.getenv("LOG_INTERVAL", 1))
        self._log_interval_unit = log_interval_unit or os.getenv(
            "LOG_INTERVAL_UNIT", "D"
        )

    def get_logger(self):
        """Return the configured logger."""
        raise NotImplementedError
