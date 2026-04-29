import regex as re
from heap import init_heap, decrement_key, push, pop
from linked_list import DoublyLinkedList

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def read_file(path: str, verbose=False):
    with open(path, 'rb') as file:
        contents = file.read() # read in utf-8 encoding
    if verbose:
        print(contents)
    return contents

def init_pair_counts(input: list):
    pair_dict = {}
    for i in range(len(input) - 1):
        val = pair_dict.get((bytes([input[i]]), bytes([input[i + 1]])), [])
        val.append(i)
        pair_dict[(bytes([input[i]]), bytes([input[i + 1]]))] = val
    pair_count_list = [(len(v), k, v) for k, v in pair_dict.items()]
    heap = init_heap(pair_count_list)
    return heap

def init_index_to_vocab(input: list):
    index_to_node_list = {}
    node_list = DoublyLinkedList()
    for i in range(len(input)):
        node = node_list.insert((i, input[i]))  # insert into tail
        index_to_node_list[i] = node
    return node_list, index_to_node_list

# vocab is updated in-place (pass by reference)
def update_pair_counts(current_counts: dict, vocab: dict, node_list: DoublyLinkedList, index_to_node_list: dict, verbose=False):
    if verbose:
        print(f'HEAP STATE: {current_counts}')
    max_pair = pop(current_counts)
    # max_pair is tuple of (num_occurrences, pair, indices)
    new_vocab = max_pair[1][0] + max_pair[1][1]
    vocab[len(vocab)] = new_vocab # add max-pair (a, b) to vocabulary
    if verbose:
        print(f'MAX PAIR: {max_pair}')
        print(f'NEW VOCAB: {new_vocab} TOKEN: {len(vocab)}')

    # maintain current counts:
    #     add occurrences of (a, b, x) and remove (b, x) for all (a, b, x).
    #     add occurrences of (x, a, b) and remove (x, a) for all (x, a, b). TODO: add previous map!
    # maintain index_to_vocab: update all occurrences of (a, b) to indices where a starts.

    merged_boundary = 0

    for i in max_pair[2]:
        if i >= merged_boundary:
            if verbose: print(f'---- ONE UPDATE ----: index {i}')
            curr_node = index_to_node_list[i]
            if verbose:
                print(f'NODE: {curr_node}')
                print(f'NEXT: {curr_node.next}')
            new_node, old_nodes = node_list.merge((i, len(vocab) - 1), curr_node)
            if verbose:
                print(f'NEW NODE: {new_node}')
                for node in old_nodes:
                    print(f'OLD NODE: {node}')
                
            # maintain index_to_node_list
            index_to_node_list[i] = new_node
            index_to_node_list.pop(old_nodes[1].data[0])

            if new_node.next is not None:
                next_vocab = vocab[new_node.next.data[1]]
                old_pair = (max_pair[1][1], bytes(next_vocab))
                if verbose: print(f'DECREMENT: {old_pair} MAX_PAIR: {max_pair[1]}')
                if old_pair != max_pair[1]: decrement_key(current_counts, old_pair, aux=[old_nodes[1].data[0]])
                new_pair = (new_vocab, next_vocab)
                if verbose: print(f'INCREMENT: {new_pair}')
                push(current_counts, (1, new_pair, [i]))
                
            if new_node.prev is not None:
                prev_vocab = vocab[new_node.prev.data[1]]
                old_pair = (bytes(prev_vocab), max_pair[1][0])
                if verbose: print(f'DECREMENT: {old_pair} MAX_PAIR: {max_pair[1]}')
                if old_pair != max_pair[1]: decrement_key(current_counts, old_pair, aux=[new_node.prev.data[0]])
                new_pair = (prev_vocab, new_vocab)
                if verbose: print(f'INCREMENT: {new_pair}')
                push(current_counts, (1, new_pair, [new_node.prev.data[0]]))

            merged_boundary = new_node.data[0] + len(vocab[new_node.data[1]])
            if verbose: print(f'MERGED BOUNDARY: {merged_boundary}')
        
    return max_pair[1]

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], verbose=False):
    file = list(read_file(input_path, verbose=verbose))
    vocab = {i: chr(i).encode('utf-8') for i in range(256)}
    
    pair_counts = init_pair_counts(file)
    node_list, index_to_node_list = init_index_to_vocab(file)

    merges = []
    while len(vocab) < vocab_size:
        merge = update_pair_counts(pair_counts, vocab, node_list, index_to_node_list, verbose=verbose)
        merges.append((merge[0], merge[1]))
    return vocab, merges

if __name__ == '__main__':
    verbose=True
    # vocab, merges = train_bpe('my_tests/sample.txt', 260, [], verbose=verbose)
    vocab, merges = train_bpe('my_tests/overlap_sample.txt', 257, [], verbose=verbose)
    print('---- VOCABULARY ----')
    for k, v in vocab.items():
        if k >= 256: print(f'{k}: {v}')
    print('---- MERGES ----')
    print(merges)
