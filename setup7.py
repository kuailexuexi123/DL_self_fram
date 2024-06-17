import numpy as np
class Variable:
    def __init__(self,data):
        self.data=data
        self.grad=None
        self.creator=None
    def set_creator(self,func):
        self.creator=func
    def backward(self):
        f=self.creator #1、获取函数
        if f is not None:
            x=f.input #2、获取函数的输入
            x.grad=f.backward(self.grad)#3、调用函数的backward方法
            x.backward()#调用前面那个变量的backward方面（递归方案）

class Function:
    def __call__(self,input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        output.set_creator(self)#让输出变量保存创造者的信息
        self.input=input #保存输入的变量
        self.output=output #保存输出变量
        return output

    def forward(self,x):
        raise NotImplementedError()

    def backward(self,x):
        raise NotImplementedError()

class Square(Function):
    def forward(self,x):
        y=x**2
        return y

    def backward(self,gy):
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self,x):
        y=np.exp(x)
        return y

    def backward(self,gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

assert y.creator==C
assert y.creator.input==b
assert y.creator.input.creator==B
assert y.creator.input.creator.input==a
assert y.creator.input.creator.input.creator==A
assert y.creator.input.creator.input.creator.input==x

###以下是实现反向传播
y.grad=np.array(1.0)
C=y.creator #1、获取函数
b=C.input #2、获取函数的输入
b.grad=C.backward(y.grad)#3、调用函数的backward方法
B=b.creator#1、获取函数
a=B.input#2、获取函数的输入
a.grad=B.backward(b.grad)#3、调用函数的backward方法
A=a.creator#1、获取函数
x=A.input#2、获取函数的输入
x.grad=A.backward(a.grad)#3、调用函数的backward方法
print(x.grad)