#encoding=utf-8  
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import math

#自定义函数 e指数形式
def func(x, a,u, sig):
    return  a*(np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig))*(431+(4750/x))


#定义x、y散点坐标
x = [1,2,3,4,5,6,7,8,9,10]
x=np.array(x)
# x = np.array(range(20))
print('x is :\n',x)
num = [278,326,547,639,916,1975,2744,4515,5974,7783]
y = np.array(num)
print('y is :\n',y)

popt, pcov = curve_fit(func, x, y,p0=[3.1,4.2,3.3])
#获取popt里面是拟合系数
a = popt[0]
u = popt[1]
sig = popt[2]


yvals = func(x,a,u,sig) #拟合y值
print(u'系数a:', a)
print(u'系数u:', u)
print(u'系数sig:', sig)

x1 = list(range(1,31))
y1 = [func(x0,a,u,sig) for x0 in x1]

#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plot3 = plt.plot(x1, y1, 'g',label='long time values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=1) #指定legend的位置右下角
plt.title('curve_fit')
plt.show()


