from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int
    is_student: bool


p1: Person = {"name": "Alice", "age": 30, "is_student": False}

p2: Person = {"name": "Bob", "age": "22", "is_student": "True"}

print(p1, p2)
