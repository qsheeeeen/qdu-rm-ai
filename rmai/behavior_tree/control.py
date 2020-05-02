from .base import ControlNode


class SequenceNode(ControlNode):
    def __init__(self, name, children):
        super(SequenceNode, self).__init__(name, children)
        self._index = 0

    def tick(self):
        while self._index < len(self.children):
            child_status = self.children[self._index].tick()

            if child_status == self.SUCCESS:
                self._index += 1
            # elif child_status == self.RUNNING:
            #     return self.RUNNING

            elif child_status == self.FAILURE:
                # HaltAllChildren()
                self._index = 0
                return self.FAILURE

        # HaltAllChildren()
        self._index = 0
        return self.SUCCESS

    def __str__(self):
        return 'SequenceNode {}'.format(self.name)


class ReactiveSequenceNode(ControlNode):
    def __init__(self, name, children):
        super(ReactiveSequenceNode, self).__init__(name, children)

    def tick(self):
        self.status = self.RUNNING

        for child in self.children:
            child_status = child.tick()

            if child_status == self.RUNNING:
                return self.RUNNING

            elif child_status == self.FAILURE:
                # HaltAllChildren()
                return self.FAILURE

        # HaltAllChildren()
        return self.SUCCESS

    def __str__(self):
        return 'ReactiveSequenceNode {}'.format(self.name)


class SequenceStar(ControlNode):
    def __init__(self, name, children):
        super(SequenceStar, self).__init__(name, children)
        self._index = 0

    def tick(self):
        while self._index < len(self.children):
            child_status = self.children[self._index].tick()

            if child_status == self.SUCCESS:
                self._index += 1

            elif child_status == self.RUNNING or child_status == self.FAILURE:
                return child_status

        # HaltAllChildren()
        self._index = 0
        return self.SUCCESS

    def __str__(self):
        return 'SequenceStar {}'.format(self.name)


class FallbackNode(ControlNode):
    def __init__(self, name, children):
        super(FallbackNode, self).__init__(name, children)
        self._index = 0

    def tick(self):
        while self._index < len(self.children):
            child_status = self.children[self._index].tick()

            if child_status == self.RUNNING:
                return self.RUNNING

            elif child_status == self.FAILURE:
                self._index += 1

            elif child_status == self.SUCCESS:
                # HaltAllChildren()
                self._index = 0
                return self.SUCCESS

        # HaltAllChildren()
        self._index = 0
        return self.FAILURE

    def __str__(self):
        return 'FallbackNode {}'.format(self.name)


class ReactiveFallbackNode(ControlNode):
    def __init__(self, name, children):
        super(ReactiveFallbackNode, self).__init__(name, children)
        self._index = 0

    def tick(self):
        while self._index < len(self.children):
            child_status = self.children[self._index].tick()

            if child_status == self.RUNNING:
                return self.RUNNING

            elif child_status == self.FAILURE:
                self._index += 1

            elif child_status == self.SUCCESS:
                # HaltAllChildren()
                return self.SUCCESS

        # HaltAllChildren()
        self._index = 0
        return self.FAILURE

    def __str__(self):
        return 'ReactiveFallbackNode {}'.format(self.name)
