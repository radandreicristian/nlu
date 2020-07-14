from typing import Any


class AbstractResultWriter:

    def get_destination(self) -> Any:
        pass

    def append_row(self, values: list, index: int) -> None:
        pass

    def save(self) -> None:
        pass
