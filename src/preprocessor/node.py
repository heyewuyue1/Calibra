class Node:
    def __init__(self, operator=None, tables=None, card=None, size_in_bytes=None) -> None:
        self.lc = None
        self.rc = None
        self.tables = tables
        self.operator = operator
        self.card = card
        self.size_in_bytes = size_in_bytes
        self.feature = None

    def __str__(self) -> str:
        return '('+ str(self.idx) + ') ' + self.operator + '\n' \
            + 'Left Child: ' + str(self.lc) + '\n'\
            + 'Right Child: ' + str(self.rc) + '\n'\
            + 'Card: ' + str(self.card) + '\n'\
            + 'Size In Bytes: ' + str(self.size_in_bytes) + '\n'\
            + 'Join Cols' + str(self.join_cols)
