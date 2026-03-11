import time
import re
from pravah.memory import SessionMemory

def benchmark():
    memory = SessionMemory(thread_id="test")
    # Add some dummy documents
    for i in range(100):
        memory.add_document(f"url_{i}", f"This is some content for document {i}. It contains some words like python, search, and memory.")

    query = "python search memory"

    # Warm up
    for _ in range(100):
        memory.search(query)

    start_time = time.perf_counter()
    iterations = 10000
    for _ in range(iterations):
        memory.search(query)
    end_time = time.perf_counter()

    print(f"Total time for {iterations} iterations: {end_time - start_time:.4f} seconds")
    print(f"Average time per iteration: {(end_time - start_time) / iterations:.8f} seconds")

if __name__ == "__main__":
    benchmark()
