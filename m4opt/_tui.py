from contextlib import contextmanager

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.text import Text

__all__ = ("progress", "status")

_progress = None
_depth = 0


@contextmanager
def progress():
    global _progress
    if _progress is None:
        with Progress(
            IndentedSpinnerColumn(finished_text="[bar.finished]✓"), TimeElapsedColumn()
        ) as new_progress:
            _progress = new_progress
            try:
                yield _progress
            finally:
                _progress = None
    else:
        yield _progress


class IndentedSpinnerColumn(SpinnerColumn):
    def render(self, task):
        return (
            Text(task.fields["depth"] * "  ")
            + (
                Text("✗", style="red")
                if task.fields["failed"]
                else super().render(task)
            )
            + Text(" ")
            + Text(task.description, style=None if task.completed else "bold")
        )


@contextmanager
def status(description: str):
    """Display nested progress bars."""
    global _depth
    with progress() as pg:
        task = pg.add_task(description, total=1, depth=_depth, failed=False)
        _depth += 1
        try:
            yield
        except:
            pg.update(task, failed=True)
            raise
        else:
            pg.update(task, advance=1, completed=True)
        finally:
            _depth -= 1


if __name__ == "__main__":
    from time import sleep

    for roman_numeral in ["I", "II", "III"]:
        with status(f"Task {roman_numeral}"):
            for letter in ["A", "B", "C"]:
                with status(f"Task {letter}"):
                    if roman_numeral == "III" and letter == "B":
                        raise RuntimeError("Failed")
                    sleep(1)
