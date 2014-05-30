import collections

class node(object):
    def __init__(self, value, children = []):
        self.value = value
        self.children = children

    def __repr__(self, level=0):
        ret = "\t"*level+repr(self.value)+"\n"
        for child in self.children:
            ret += child.__repr__(level+1)
        return ret

# Tree of dicts
    def Tree():
        return collections.defaultdict(Tree)
