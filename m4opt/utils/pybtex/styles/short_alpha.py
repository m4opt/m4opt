"""Pybtex style to display at most 3 authors in a reference."""

from pybtex.database import Person
from pybtex.plugin import register_plugin
from pybtex.style.formatting.alpha import Style as AlphaStyle
from pybtex.style.template import FieldIsMissing, join, node, sentence

max_names = 3


@node
def short_names(children, context, role):
    """Return formatted names."""

    assert not children

    try:
        persons = context["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context["entry"])

    style = context["style"]
    if len(persons) < max_names:
        formatted_names = [
            style.format_name(person, style.abbreviate_names) for person in persons
        ]
        return join(sep=", ", sep2=" and ", last_sep=", and ")[
            formatted_names
        ].format_data(context)
    else:
        formatted_names = [
            style.format_name(person, style.abbreviate_names)
            for person in [*persons[:max_names], Person("et al.")]
        ]
        return join(sep=", ")[formatted_names].format_data(context)


class ShortAlphaStyle(AlphaStyle):
    def format_names(self, role, as_sentence=True):
        formatted_names = short_names(role)
        if as_sentence:
            return sentence[formatted_names]
        else:
            return formatted_names


register_plugin("pybtex.style.formatting", "short_alpha", ShortAlphaStyle)
