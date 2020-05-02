from .base import BehaviorTreeNode


class SyncActionNode(BehaviorTreeNode):
    def __init__(self, name):
        super(SyncActionNode, self).__init__(name)

    def tick(self):
        raise NotImplementedError

    def __str__(self):
        return 'SyncActionNode {}'.format(self.name)


class AsyncActionNode(BehaviorTreeNode):
    def tick(self):
        # TODO: Support async.
        raise NotImplemented

    def __str__(self):
        return 'AsyncActionNode {}'.format(self.name)


# TODO: Unfinished.
class ConditionNode(BehaviorTreeNode):
    def tick(self):
        raise NotImplementedError

    def __str__(self):
        return 'ConditionNode {}'.format(self.name)
