from functools import wraps
from time import time


class bench:
    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        duration = time() - self.start
        print(f"bench: {duration:4f}s")

    def __call__(self, func=None):
        name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            duration = time() - start
            print(f"bench: '{name}' {duration:4f}s")

            return result

        return wrapper


if __name__ == "__main__":
    with bench():
        print("Hello world!")

    @bench()
    def foo():
        print("Hello world!")

    foo()
