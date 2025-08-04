from abc import ABC, abstractmethod


class Parent(ABC):
    @abstractmethod
    def __call__(self, x):
        ...

class Child(Parent):
    def __call__(self, x):
        print(x)


a = Child()
a(3)
