from multiprocessing import Queue, Value, Lock

class TrackedQueue:
    """A queue wrapper that tracks its own size using shared memory."""

    def __init__(self, name="Queue", maxsize=0):
        self.name = name
        self.queue = Queue(maxsize=maxsize)
        self._size = Value("i", 0)
        self._mp_lock = Lock()

    def put(self, item, block=True, timeout=None):
        """Add an item to the queue and increment the size counter."""

        self.queue.put(item, block=block, timeout=timeout)

        with self._mp_lock:
            self._size.value += 1

    def get(self, block=True, timeout=None):
        """Get an item from the queue and decrement the size counter."""
        
        item = self.queue.get(block=block, timeout=timeout)

        with self._mp_lock:
            if self._size.value > 0:
                self._size.value -= 1

        return item

    def size(self):
        """Get the current size of the queue."""
        with self._mp_lock:
            return self._size.value
