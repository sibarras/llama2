class Foo:
    name: str

    def __init__(self, name: str):
        self.name = name
    
    @property
    def greet(self) -> str:
        return f"Hola, {self.name}!"