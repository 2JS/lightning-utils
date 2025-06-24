from functools import wraps
from time import time


class bench:
  def __init__(self, description=None):
    self.description = description

  def __enter__(self):
    self.start = time()
    return self

  def __exit__(self, *args):
    duration = time() - self.start
    print(f"bench: '{self.description}' {duration:4f}s")

  def __call__(self, func=None):
    description = self.description or func.__name__

    @wraps(func)
    def wrapper(*args, **kwargs):
      start = time()
      result = func(*args, **kwargs)
      duration = time() - start
      print(f"bench: '{description}' {duration:4f}s")

      return result

    return wrapper


if __name__ == "__main__":
  with bench("hello"):
    print("Hello world!")

  @bench()
  def foo():
    print("Hello world!")

  foo()
