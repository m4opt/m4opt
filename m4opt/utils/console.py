"""Utilities for console applications and text user interfaces.

Use the :meth:`progress` and :meth:`status` methods to create live feedback
for a nested series of tasks. The elapsed time is shown for each task, along
with whether it completed successfully or failed due to an exception being
raised.

Examples
--------

.. code:: python

    from m4opt.utils.console import progress, status
    from time import sleep
    with progress():
        with status("Prepare the dough"):
            with status("Proof the yeast"):
                pass  # do some work here
            with status("Mix the wet and dry ingredients"):
                pass  # do some work here
            with status("Knead for 10 minutes"):
                pass  # do some work here
            with status("Let rise for 1 hour"):
                pass  # do some work here
        with status("Preheat oven to 500° F"):
            pass  # do some work here
        with status("Assemble the pizza"):
            with status("Roll out the dough"):
                pass  # do some work here
            with status("Top with sauce and cheese"):
                pass  # do some work here
            with status("Add your favorite toppings"):
                pass  # do some work here
        with status("Bake until golden brown"):
            pass
        with status("Serve to your hungry guests"):
            raise RuntimeError("Sorry, I ate it all")

.. code:: text

    ✓ Prepare the dough                 0:00:00
      ✓ Proof the yeast                 0:00:00
      ✓ Mix the wet and dry ingredients 0:00:00
      ✓ Knead for 10 minutes            0:00:00
      ✓ Let rise for 1 hour             0:00:00
    ✓ Preheat oven to 500° F            0:00:00
    ✓ Assemble the pizza                0:00:00
      ✓ Roll out the dough              0:00:00
      ✓ Top with sauce and cheese       0:00:00
      ✓ Add your favorite toppings      0:00:00
    ✓ Bake until golden brown           0:00:00
    ✗ Serve to your hungry guests       0:00:00
    Traceback (most recent call last):
      File "/Users/lpsinger/src/m4opt/test.py", line 24, in <module>
        raise RuntimeError("Sorry, I ate it all")
    RuntimeError: Sorry, I ate it all
"""

from contextlib import contextmanager

import rich.console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.text import Text

__all__ = ("progress", "status")

_progress = None
_depth = 0
_max_depth = 2
_is_jupyter = rich.console._is_jupyter()


@contextmanager
def progress():
    """Context manager to create a live display for showing status of tasks.

    If there is already an active progress display, this method will return it
    instead of creating a new one.
    """
    global _progress
    if not _is_jupyter and _progress is None:
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
            + Text(f" {task.description}", style=None if task.completed else "bold")
        )


@contextmanager
def status(description: str):
    """Context manager to track the runtime of a task."""
    global _depth
    if _is_jupyter or _depth >= _max_depth:
        yield
    else:
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
                    for number in ["1", "2", "3"]:
                        if roman_numeral == "III" and letter == "B" and number == "1":
                            raise RuntimeError("Failed")
                        sleep(1)
