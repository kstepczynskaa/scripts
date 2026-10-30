from decimal import Decimal


from .nodes import Node
from .edges import Edge


class Intron:
    """
    A manually supplied intron.

    Coordinates are 1-based and inclusive.
    This initial version supports forward-strand introns only.
    """

    def __init__(self, start, stop, frame=1):
        self.start = start
        self.stop = stop
        self.frame = frame


    def nodes(self):
        start_node = Node(
            "intron",
            "start",

            self.frame,
            self.start
        )


        stop_node = Node(
            "intron",
            "stop",
            self.frame,
            self.stop
        )

        return start_node, stop_node

    def weight(self, sequence):
        """
        Initial experimental intron score.

        Canonical GT...AG introns receive -20.
        Other introns receive +20.
        """
        intron_sequence = sequence[self.start - 1:self.stop].lower()

        if (
            intron_sequence.startswith("gt")
            and intron_sequence.endswith("ag")
        ):
            return Decimal("-20.0")

        return Decimal("20.0")
