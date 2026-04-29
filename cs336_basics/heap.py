# Initialize Heap from List
# Input: List of (Key, Identifier, Auxiliary Information)
# Build Max-Heap sorted by key (tie-breaking by Identifier) in place.
def init_heap(input: list):
    heap = input.copy()
    for i in range(len(heap) - 1, -1, -1):
        _sift_down_no_id(heap, i)

    id_map = {}
    for i in range(len(heap)):
        id_map[heap[i][1]] = i

    id_heap = {'heap': heap, 'id_map': id_map}
    return id_heap

# Given identifier, decrease key and maintain heap invariant (sift down).
def decrement_key(id_heap, id, amt=1, aux=[]):
    print('---- DECREMENT KEY ----')
    print(id, aux)
    heap = id_heap['heap']
    id_map = id_heap['id_map']
    new_val = list(heap[id_map[id]])

    if aux:
        new_val[2] = [x for x in new_val[2] if x not in aux]
        new_val[0] = len(new_val[2])
    else:
        new_val[0] -= amt
    
    if new_val[0] == 0:
        remove_key(id_heap, id)
    else:
        heap[id_map[id]] = tuple(new_val)
        # print(heap[id_map[id]])
        _sift_down(id_heap, id_map[id])

def increment_key(id_heap, id, amt=1, aux=[]):
    heap = id_heap['heap']
    id_map = id_heap['id_map']
    new_val = list(heap[id_map[id]])
    if aux:
        new_val[2] += aux
        new_val[0] = len(new_val[2])
    else:
        new_val[0] += amt
    heap[id_map[id]] = tuple(new_val)
    _sift_up(id_heap, id_map[id])

def remove_key(id_heap, id):
    heap = id_heap['heap']
    id_map = id_heap['id_map']
    if id in id_map:
        index = id_map[id]
        heap[index] = heap.pop()
        id_map[heap[index][1]] = index
        _sift_down(id_heap, index)
        _sift_up(id_heap, index)
        id_map.pop(id)

# Return maximum element
def pop(id_heap):
    heap = id_heap['heap']
    id_map = id_heap['id_map']
    if len(heap) == 0: 
        return None
    if len(heap) == 1: 
        id_map = {}
        return heap.pop()
    
    root = heap[0]
    id_map.pop(root[1])
    heap[0] = heap.pop()
    id_map[heap[0][1]] = 0
    _sift_down(id_heap, 0)
    return root

# Add element to heap and maintain heap invariant.
def push(id_heap, x):
    heap = id_heap['heap']
    id_map = id_heap['id_map']
    if x[1] in id_map:
        increment_key(id_heap, x[1], aux=x[2])
    else:
        heap.append(x)
        id_map[x[1]] = len(heap) - 1
        _sift_up(id_heap, len(heap) - 1)

def _sift_up(id_heap, i):
    heap = id_heap['heap']

    parent = (i - 1) // 2
    if i > 0 and heap[i] > heap[parent]:
        _swap_items(id_heap, i, parent)
        _sift_up(id_heap, parent)

def _swap_items(id_heap, i, j):
    heap = id_heap['heap']
    id_map = id_heap['id_map']

    id_map[heap[i][1]], id_map[heap[j][1]] = j, i
    heap[i], heap[j] = heap[j], heap[i]

def _sift_down(id_heap, i):
    heap = id_heap['heap']
    left, right = 2 * i + 1, 2 * i + 2
    largest = i
    if left < len(heap) and heap[left] > heap[largest]:
        largest = left
    if right < len(heap) and heap[right] > heap[largest]:
        largest = right
    if largest != i:
        _swap_items(id_heap, i, largest)
        _sift_down(id_heap, largest)

def _sift_down_no_id(heap, i):
    left, right = 2 * i + 1, 2 * i + 2
    largest = i
    if left < len(heap) and heap[left] > heap[largest]:
        largest = left
    if right < len(heap) and heap[right] > heap[largest]:
        largest = right
    if largest != i:
        heap[i], heap[largest] = heap[largest], heap[i]
        _sift_down_no_id(heap, largest)

if __name__ == '__main__':
    test = [(1, 'chris'), (7, 'max'), (4, 'kasey'), (10, 'sharon'), (3, 'ahmi'), (6, 'keaton')]
    print(test)
    id_heap = init_heap(test)
    print(id_heap)
    decrement_key(id_heap, 'sharon', amt=4)
    print(id_heap)
    print(pop(id_heap))
    print(id_heap)