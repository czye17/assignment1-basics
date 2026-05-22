import os
import pdb
import pickle
import regex as re

from collections import defaultdict
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import heapq

from cs336_basics.tokenizer import get_counts, merge
from cs336_basics.reverse import ReverseOrder


""" Data Structures
    word -> count           : static dictionary of words (from pretokenization) and how often they occur
    word -> tokens          : representation of each word as current list of tokens
    word -> pair_cnts       : pair counts of tokens representing word
    pair -> count           : number of times each pair occurs (over all words) -- dictionary and heap
    pair -> set[word]       : words in which the pairs occur
"""


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BASE_VOCAB = 256


def get_chunk_boundaries(file, desired_num_chunks: int, special_tokens: list[bytes]):
    for tok in special_tokens:
        assert isinstance(tok, bytes)

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        # pdb.set_trace()
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_flag = False
            # Find the special token in the mini chunk
            for tok in special_tokens:
                found_at = mini_chunk.find(tok)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    found_flag = True
                    break

            if found_flag:
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_special_tokens(input: str, special_tokens: list, verbose=False):
    escaped_special_tokens = [re.escape(x) for x in special_tokens]
    delimiter = '|'.join(escaped_special_tokens)
    if verbose: print(f'Split Delimiter: {delimiter}')
    split_str = re.split(delimiter, input)
    return split_str


def test_special_tokens():
    print('---- TEST: Split Special Tokens ----')
    delimiters = ['<|endoftext|>', '<|endofprompt|>']
    sample_str = 'abc<|endoftext|>def<|endofprompt|>xyz'
    test_output = split_special_tokens(sample_str, delimiters, verbose=True)
    output = ['abc', 'def', 'xyz']
    print(f'Output Match: {test_output == output}')


def pretokenize(inputs: list[str], verbose=False):
    iterators = [re.finditer(PAT, x) for x in inputs]
    return iterators


def init_counts(docs, verbose=False):
    word_counts = defaultdict(int)
    occurrences = defaultdict(set)
    word_to_tokens = {}
    word_to_pair_cnts = {}
    
    for doc in docs:
        for match in doc:
            word = match.group().encode('utf-8')
            if word not in word_counts:
                if verbose: print(f'Initializing Word: {word}')
                assert(word not in word_to_tokens)
                assert(word not in word_to_pair_cnts)
                tokens = list(word)
                word_to_tokens[word] = tokens
                word_to_pair_cnts[word] = defaultdict(int)
                if len(tokens) > 1:
                    for i in range(len(tokens) - 1):
                        pair = (tokens[i], tokens[i + 1])
                        word_to_pair_cnts[word][pair] += 1
                        occurrences[pair].add(word)
            word_counts[word] += 1
        
    return word_counts, word_to_tokens, word_to_pair_cnts, occurrences


# given pair, return representation in bytes
def pair_to_comparator(pair, vocab):
    p0, p1 = pair
    # token -> byte string -> reverse
    c0 = ReverseOrder(vocab[p0])
    c1 = ReverseOrder(vocab[p1])
    return (c0, c1)


def comparator_to_pair(c_pair, rev_vocab):
    c0, c1 = c_pair
    # reverse -> byte string -> token
    p0 = rev_vocab[bytes(c0.get_val())]
    p1 = rev_vocab[bytes(c1.get_val())]
    return (p0, p1)


def initialize_bpe_thread(bounds, input_path, special_tokens, verbose=False):
    start, end = bounds
    with open(input_path, 'rb') as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
        # Pretokenization
    subchunks = split_special_tokens(chunk, special_tokens, verbose=verbose)
    docs = pretokenize(subchunks, verbose=verbose)
    word_counts, word_to_tokens, word_to_pair_cnts, pair_occurrences = init_counts(docs, verbose=verbose)

    return word_counts, word_to_tokens, word_to_pair_cnts, pair_occurrences
    # c_vocab, c_rev_vocab, c_word_counts, c_word_to_tokens, c_word_to_pair_cnts = initialize_bpe(docs, verbose=verbose)


def add_to_int_default_dict(main: defaultdict[int], other: defaultdict[int]):
    for k, v in other.items():
        main[k] += v


def add_to_set_default_dict(main: defaultdict[set], other: defaultdict[set]):
    for k, v in other.items():
        main[k].update(v)


def insert_to_dict(main: dict, other: dict):
    for k, v in other.items():
        if k not in main:
            main[k] = v


def initialize_bpe_parallel(input_path, special_tokens, parallel=False, verbose=False, num_chunks=8):
    special_tokens_bytes = [tok.encode('utf-8') for tok in special_tokens]
    vocab = {i: bytes([i]) for i in range(BASE_VOCAB)}
    rev_vocab = {val: tok for tok, val in vocab.items()}

    with open(input_path, 'rb') as file:
        chunk_boundaries = get_chunk_boundaries(file, desired_num_chunks=num_chunks, special_tokens=special_tokens_bytes)
        list_bounds = zip(chunk_boundaries[:-1], chunk_boundaries[1:])

    word_counts = defaultdict(int)
    pair_occurrences = defaultdict(set)
    word_to_tokens = {}
    word_to_pair_cnts = {}
    if parallel:
        single_thread_func = partial(initialize_bpe_thread, input_path=input_path, special_tokens=special_tokens, verbose=verbose)

        # CPU-bound → ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(single_thread_func, list_bounds))
            for result in results:
                c_word_counts, c_word_to_tokens, c_word_to_pair_cnts, c_pair_occurrences = result
                add_to_int_default_dict(word_counts, c_word_counts)
                add_to_set_default_dict(pair_occurrences, c_pair_occurrences)
                insert_to_dict(word_to_tokens, c_word_to_tokens)
                insert_to_dict(word_to_pair_cnts, c_word_to_pair_cnts)

        # I/O-bound → ThreadPoolExecutor
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #     futures = [executor.submit(single_thread_func, bounds) for bounds in list_bounds]
        #     for future in as_completed(futures):
        #         # complete = futures[future]           # look up which URL this was
        #         try:
        #             result = future.result()
        #             print(f"✓ Got one result.")   # process immediately
        #             c_word_counts, c_word_to_tokens, c_word_to_pair_cnts, c_pair_occurrences = result
        #             add_to_int_default_dict(word_counts, c_word_counts)
        #             add_to_set_default_dict(pair_occurrences, c_pair_occurrences)
        #             insert_to_dict(word_to_tokens, c_word_to_tokens)
        #             insert_to_dict(word_to_pair_cnts, c_word_to_pair_cnts)
        #         except Exception as e:
        #             print(f"✗ failed.")
                
        # raise NotImplementedError('Parallel BPE Initialization Not Implemented.')
    else:
        for (start, end) in list_bounds:
            c_word_counts, c_word_to_tokens, c_word_to_pair_cnts, c_pair_occurrences = initialize_bpe_thread((start, end), input_path, special_tokens, verbose=verbose)
            add_to_int_default_dict(word_counts, c_word_counts)
            add_to_set_default_dict(pair_occurrences, c_pair_occurrences)
            insert_to_dict(word_to_tokens, c_word_to_tokens)
            insert_to_dict(word_to_pair_cnts, c_word_to_pair_cnts)
    
    pair_counts = defaultdict(int)
    for pair, word_set in pair_occurrences.items():
        for word in word_set:
            pair_counts[pair] += (word_counts[word] * word_to_pair_cnts[word][pair])

    heap_counts = []
    for pair, cnt in pair_counts.items():
        heapq.heappush(heap_counts, (-cnt, pair_to_comparator(pair, vocab)))

    return vocab, rev_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences


def check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word):
    word_pair_cnts = get_counts(word_to_tokens[word])
    for pair in word_pair_cnts:
        total_count = pair_counts[pair]
        aggregate_count = sum([word_to_pair_cnts[w][pair] * word_counts[w] for w in pair_occurrences[pair]])
        if total_count != aggregate_count: 
            pdb.set_trace()
            print(f'Count Mismatch for Pair {pair}')
            return False
    return True


def train_merge(best_pair, new_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, pair_occurrences):
    words = pair_occurrences[best_pair].copy()   
    updated_pairs = set()
    for word in words:
        # debug_counts = get_counts(word_to_tokens[word])
        # pdb.set_trace()
        # assert(check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word)) 
        for pair, cnt in word_to_pair_cnts[word].items(): 
            # assert(cnt == debug_counts[pair])
            pair_counts[pair] -= (word_counts[word] * cnt)
            pair_occurrences[pair].remove(word)
            updated_pairs.add(pair)
        
        word_to_pair_cnts[word] = defaultdict(int)
        # assert(check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word))

        old_tokens = word_to_tokens[word]
        word_to_tokens[word] = merge(best_pair, new_vocab, old_tokens)
        if len(word_to_tokens[word]) > 1: 
            word_to_pair_cnts[word] = get_counts(word_to_tokens[word])
            for pair, cnt in word_to_pair_cnts[word].items():
                pair_counts[pair] += (word_counts[word] * cnt)
                pair_occurrences[pair].add(word)
                updated_pairs.add(pair)
            # assert(check_consistency_word_to_pair_cnts_and_pair_cnts(word_counts, word_to_pair_cnts, pair_counts, pair_occurrences, word_to_tokens, word))
    
    return updated_pairs


def get_best_pair(pair_counts, heap_counts, rev_vocab, verbose=False):
    best_count, neg_best_pair = heapq.heappop(heap_counts)
    best_pair = comparator_to_pair(neg_best_pair, rev_vocab)
    while pair_counts[best_pair] + best_count != 0:
        best_count, neg_best_pair = heapq.heappop(heap_counts)
        best_pair = comparator_to_pair(neg_best_pair, rev_vocab)
    
    best_count = - best_count
    return best_count, best_pair


def train_bpe_parallel(input_path: str, vocab_size: int, special_tokens: list[str], num_chunks=8, parallel=False, verbose=False):
    vocab, rev_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, heap_counts, pair_occurrences = initialize_bpe_parallel(input_path, special_tokens, num_chunks=num_chunks, parallel=parallel, verbose=verbose)
    merges = []

    while len(vocab) < vocab_size - len(special_tokens):
        best_count, best_pair = get_best_pair(pair_counts, heap_counts, rev_vocab, verbose=verbose)
        p0, p1 = best_pair
        if best_count <= 1: 
            if verbose: print(f'No pairs with occurrence > 1')
            break
        if verbose:
            print(f'Merging {best_pair} {(vocab[p0], vocab[p1])} with value {best_count} to {len(vocab)}')

        new_vocab = len(vocab)
        vocab[new_vocab] = vocab[p0] + vocab[p1]
        rev_vocab[vocab[p0] + vocab[p1]] = new_vocab
        merges.append((vocab[p0], vocab[p1]))
        updated_pairs = train_merge(best_pair, new_vocab, word_counts, word_to_tokens, word_to_pair_cnts, pair_counts, pair_occurrences)
        for pair in updated_pairs:
            heapq.heappush(heap_counts, (-pair_counts[pair], pair_to_comparator(pair, vocab)))

    for token in special_tokens:
        new_vocab = len(vocab)
        vocab[new_vocab] = token.encode('utf-8')
    
    return vocab, merges


def basic_test():
    test_special_tokens()
    print('---- TEST: Train BPE ----')
    vocab, merges = train_bpe_parallel('my_tests/sample.txt', 356, ['<|endoftext|>'], num_chunks=1, verbose=False)
    print(f'VOCABULARY: {vocab}')
    print(f'MERGES: {merges}')


def train_bpe_tokenizer(file_path, vocab_size, dest):
    vocab, merges = train_bpe_parallel(file_path, vocab_size, ['<|endoftext|>'], num_chunks=10, verbose=False)

    tokenizer_data = {
        'vocab': vocab,
        'merges': merges
    }

    with open(dest, 'w') as f:
        pickle.dump(tokenizer_data, f)


if __name__ == '__main__':
    # basic_test()
    train_bpe_tokenizer('data/TinyStoriesV2-GPT4-train.txt', 10000, 'my_tests/TinyStoriesTokenizer.pkl')

