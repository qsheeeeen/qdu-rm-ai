class BehaviorTreeNode(object):
    SUCCESS = True
    FAILURE = False
    RUNNING = 'RUNNING'

    def __init__(self, name):
        self.status = None
        self.name = name

    def tick(self):
        raise NotImplemented


class ControlNode(BehaviorTreeNode):
    def __init__(self, name, children):
        super(ControlNode, self).__init__(name)
        self.children = children

    def tick(self):
        raise NotImplementedError


class DecoratorNode(BehaviorTreeNode):
    def __init__(self, name, child):
        super(DecoratorNode, self).__init__(name)
        self.child = child

    def tick(self):
        raise NotImplementedError
