class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def insert(self, data, node=None):
        new_node = Node(data)

        if node:
            new_node.next = node.next
            new_node.prev = node

            if node.next is not None:
                node.next.prev = new_node
            node.next = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

        return new_node
    
    def merge(self, data, curr_node):
        # if node is b, then a <-> b <-> c <-> d becomes a <-> new_node <-> d
        assert(curr_node is not None and curr_node.next is not None)
        self.delete(curr_node.next) # a <-> b <-> d
        new_node = self.insert(data, node=curr_node) # a <-> b <-> new_node <-> d
        self.delete(curr_node) # a <-> new_node <-> d
        return new_node
    
    def delete(self, node):
        if node.next is None and node.prev is None: # single node
            self.head = None
            self.tail = None
        elif node.next is None: # node is tail
            node.prev.next = None
            self.tail = node.prev
        elif node.prev is None: # node is head
            node.next.prev = None
            self.head = node.next
        else: # node is internal
            node.next.prev = node.prev
            node.prev.next = node.next
