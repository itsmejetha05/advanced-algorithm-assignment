"""
Task 1: Advanced Data Structures for a Route Planning Application
===================================================================

Implements four data structures used to store / retrieve city records:

    1. Binary Search Tree (BST)      - keyed by city name
    2. AVL Tree (self-balancing BST) - keyed by city name
    3. Min-Heap                      - keyed by distance (priority queue)
    4. Hash Table (separate chaining)- keyed by city name

Each structure exposes: insert(key, data), search(key), delete(key)
so that the benchmarking harness can drive them identically.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 1. Binary Search Tree (unbalanced)
# ---------------------------------------------------------------------------
class BSTNode:
    __slots__ = ("key", "data", "left", "right")

    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.left = None
        self.right = None


class BinarySearchTree:
    """Classic, *unbalanced* BST. Degrades to O(n) on sorted input."""

    def __init__(self):
        self.root = None
        self._size = 0

    def insert(self, key, data=None):
        self._size += 1
        if self.root is None:
            self.root = BSTNode(key, data)
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = BSTNode(key, data)
                    return
                node = node.left
            elif key > node.key:
                if node.right is None:
                    node.right = BSTNode(key, data)
                    return
                node = node.right
            else:
                node.data = data  # update
                self._size -= 1
                return

    def search(self, key):
        node = self.root
        while node is not None:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node.data
        return None

    def delete(self, key):
        self.root, deleted = self._delete(self.root, key)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, key):
        if node is None:
            return node, False
        if key < node.key:
            node.left, deleted = self._delete(node.left, key)
        elif key > node.key:
            node.right, deleted = self._delete(node.right, key)
        else:
            deleted = True
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True
            # two children: replace with in-order successor
            succ = node.right
            while succ.left is not None:
                succ = succ.left
            node.key, node.data = succ.key, succ.data
            node.right, _ = self._delete(node.right, succ.key)
        return node, deleted

    def height(self):
        """Iterative height computation (avoids recursion-depth issues on
        the pathological, list-like tree produced by sorted insertion)."""
        if self.root is None:
            return 0
        max_h = 0
        stack = [(self.root, 1)]
        while stack:
            node, depth = stack.pop()
            max_h = max(max_h, depth)
            if node.left:
                stack.append((node.left, depth + 1))
            if node.right:
                stack.append((node.right, depth + 1))
        return max_h

    def __len__(self):
        return self._size


# ---------------------------------------------------------------------------
# 2. AVL Tree (self-balancing BST)
# ---------------------------------------------------------------------------
class AVLNode:
    __slots__ = ("key", "data", "left", "right", "height")

    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """Self-balancing BST guaranteeing O(log n) height at all times."""

    def __init__(self):
        self.root = None
        self._size = 0

    @staticmethod
    def _h(node):
        return node.height if node else 0

    def _update(self, node):
        node.height = 1 + max(self._h(node.left), self._h(node.right))

    def _balance(self, node):
        return self._h(node.left) - self._h(node.right)

    def _rotate_right(self, y):
        x = y.left
        y.left = x.right
        x.right = y
        self._update(y)
        self._update(x)
        return x

    def _rotate_left(self, x):
        y = x.right
        x.right = y.left
        y.left = x
        self._update(x)
        self._update(y)
        return y

    def _rebalance(self, node):
        self._update(node)
        balance = self._balance(node)
        if balance > 1:
            if self._balance(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1:
            if self._balance(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def insert(self, key, data=None):
        self.root, inserted = self._insert(self.root, key, data)
        if inserted:
            self._size += 1

    def _insert(self, node, key, data):
        if node is None:
            return AVLNode(key, data), True
        if key < node.key:
            node.left, inserted = self._insert(node.left, key, data)
        elif key > node.key:
            node.right, inserted = self._insert(node.right, key, data)
        else:
            node.data = data
            return node, False
        return self._rebalance(node), inserted

    def search(self, key):
        node = self.root
        while node is not None:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node.data
        return None

    def delete(self, key):
        self.root, deleted = self._delete(self.root, key)
        if deleted:
            self._size -= 1
        return deleted

    def _delete(self, node, key):
        if node is None:
            return node, False
        if key < node.key:
            node.left, deleted = self._delete(node.left, key)
        elif key > node.key:
            node.right, deleted = self._delete(node.right, key)
        else:
            deleted = True
            if node.left is None:
                return node.right, True
            if node.right is None:
                return node.left, True
            succ = node.right
            while succ.left is not None:
                succ = succ.left
            node.key, node.data = succ.key, succ.data
            node.right, _ = self._delete(node.right, succ.key)
        return self._rebalance(node), deleted

    def height(self):
        return self._h(self.root)

    def __len__(self):
        return self._size


# ---------------------------------------------------------------------------
# 3. Min-Heap (array based binary heap) - priority queue
# ---------------------------------------------------------------------------
class MinHeap:
    """
    Binary min-heap keyed on a priority value (e.g. distance to next city).
    Stores (priority, data) pairs. Supports insert (push) and
    extract_min (pop) in O(log n), peek in O(1).
    """

    def __init__(self):
        self._heap = []  # list of [priority, seq, data]
        self._seq = 0     # tie-breaker to keep heap stable / comparable

    def __len__(self):
        return len(self._heap)

    def insert(self, priority, data=None):
        self._seq += 1
        entry = [priority, self._seq, data]
        self._heap.append(entry)
        self._sift_up(len(self._heap) - 1)

    def peek(self):
        if not self._heap:
            return None
        return self._heap[0][0], self._heap[0][2]

    def extract_min(self):
        if not self._heap:
            return None
        root = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return root[0], root[2]

    def _sift_up(self, i):
        heap = self._heap
        while i > 0:
            parent = (i - 1) // 2
            if heap[i][0] < heap[parent][0]:
                heap[i], heap[parent] = heap[parent], heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        heap = self._heap
        n = len(heap)
        while True:
            left, right = 2 * i + 1, 2 * i + 2
            smallest = i
            if left < n and heap[left][0] < heap[smallest][0]:
                smallest = left
            if right < n and heap[right][0] < heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            heap[i], heap[smallest] = heap[smallest], heap[i]
            i = smallest


# ---------------------------------------------------------------------------
# 4. Hash Table (separate chaining, dynamic resize)
# ---------------------------------------------------------------------------
class HashTable:
    """
    Hash table using separate chaining for collision resolution.
    Resizes (doubles) when the load factor exceeds 0.75, keeping
    average-case O(1) insert / search / delete.
    """

    def __init__(self, capacity=16):
        self._capacity = capacity
        self._buckets = [[] for _ in range(capacity)]
        self._size = 0

    def _hash(self, key):
        return hash(key) % self._capacity

    def _resize(self):
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [[] for _ in range(self._capacity)]
        for bucket in old_buckets:
            for k, v in bucket:
                idx = self._hash(k)
                self._buckets[idx].append((k, v))

    def insert(self, key, data=None):
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, data)
                return
        bucket.append((key, data))
        self._size += 1
        if self._size / self._capacity > 0.75:
            self._resize()

    def search(self, key):
        idx = self._hash(key)
        for k, v in self._buckets[idx]:
            if k == key:
                return v
        return None

    def delete(self, key):
        idx = self._hash(key)
        bucket = self._buckets[idx]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def __len__(self):
        return self._size

    def load_factor(self):
        return self._size / self._capacity
