from typing import Union, List

Numberable = Union [float, int]

def ensure_number(num: Numberable) -> 'NumberWithGrad':
    if isinstance(num, NumberWithGrad):
        return num
    else:
        return NumberWithGrad(num)


class NumberWithGrad(object):
    def __init__(self,
                 num: Numberable,
                 depends_on: list[Numberable] = None,
                 creation_op: str = ''):
        self.num = num
        self.grad = None
        self.depends_on = depends_on or []
        self.creation_op = creation_op
    
    def __add__(self,
                other: Numberable) -> 'NumberWithGrad':
        return NumberWithGrad(self.num + ensure_number(other).num,
                              depends_on=[self, ensure_number(other)],
                              creation_op='add')
    
    def __mul__(self,
                other: Numberable) -> 'NumberWithGrad':
        return NumberWithGrad(self.num * ensure_number(other).num,
                              depends_on=[self, ensure_number(other)],
                              creation_op='mul')
    
    def backward(self, backward_grad: Numberable = None) -> None:
        if backward_grad == None:
            self.grad = 1
        else:
            if self.grad == None:
                self.grad = backward_grad
            else:
                self.grad += backward_grad

        if self.creation_op == 'add':
            for dependency in self.depends_on:
                dependency.backward(self.grad)
        if self.creation_op == 'mul':
            assert len(self.depends_on) == 2
            new = self.depends_on[1] * self.grad
            self.depends_on[0].backward(new.num)
            new = self.depends_on[0] * self.grad
            self.depends_on[1].backward(new.num)
    




if __name__ == "__main__":
    a = NumberWithGrad(3)
    b = a * 4
    c = b + 3
    d = c * (a + 2)
    d.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)