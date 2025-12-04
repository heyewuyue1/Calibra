class Node:
    def __init__(self, operator=None, executed=0, tables=None, card=None, size_in_bytes=None, data={}) -> None:
        self.lc = None
        self.rc = None
        self.tables = tables
        self.operator = operator
        self.executed = executed
        self.card = card
        self.size_in_bytes = size_in_bytes
        self.feature = None
        self.data = data
