from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):
    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr  # EmailStr = "acb@gmail.com" -> to set default
    cgpa: float = Field(
        default=5.0,
        gt=0,
        lt=10,
        description="Decimal value representing the CGPA of student",
    )


new_student = {"age": 32, "email": "abc@def.com", "cgpa": "8.5"}

student = Student(**new_student)

print(student)
print(type(student))
print(student.name)

student_dict = dict(student)  # Convert to dict
print(student_dict)

student_json = student.model_dump_json()  # Convert to JSON
print(student_json)
