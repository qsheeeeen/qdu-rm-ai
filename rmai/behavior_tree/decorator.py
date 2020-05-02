from .base import DecoratorNode


class InverterNode(DecoratorNode):
    def tick(self):
        child_status = self.child.tick()

        if child_status == self.RUNNING:
            return self.RUNNING

        elif child_status == self.FAILURE:
            return self.SUCCESS

        elif child_status == self.SUCCESS:
            return self.FAILURE

    def __str__(self):
        return 'InverterNode {}'.format(self.name)


class ForceSuccessNode(DecoratorNode):
    def tick(self):
        self.child.tick()
        return self.SUCCESS

    def __str__(self):
        return 'ForceSuccessNode {}'.format(self.name)


class ForceFailureNode(DecoratorNode):
    def tick(self):
        self.child.tick()
        return self.FAILURE

    def __str__(self):
        return 'ForceFailureNode {}'.format(self.name)


class RepeatNode(DecoratorNode):
    def __init__(self, name, child, n):
        super(RepeatNode, self).__init__(name, child)
        self._n = n

    def tick(self):
        for i in range(self._n):
            child_status = self.child.tick()

            if child_status == self.RUNNING:
                return self.RUNNING

            elif child_status == self.FAILURE:
                return self.FAILURE

        return self.SUCCESS

    def __str__(self):
        return 'RepeatNode {}'.format(self.name)


class RetryNode(DecoratorNode):
    def __init__(self, name, child, n):
        super(RetryNode, self).__init__(name, child)
        self._n = n

    def tick(self):
        for i in range(self._n):
            child_status = self.child.tick()

            if child_status == self.RUNNING:
                return self.RUNNING

            elif child_status == self.SUCCESS:
                return self.SUCCESS

        return self.FAILURE

    def __str__(self):
        return 'RetryNode {}'.format(self.name)
