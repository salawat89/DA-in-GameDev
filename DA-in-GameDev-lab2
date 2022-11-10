# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Ахмадиев Салават Русланович
- РИ211102
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Ход работы: Написать программу Hello World на Python и Unity
1. Python. Использовал среду разработки PyCharm.

![image](https://user-images.githubusercontent.com/114305772/192601735-5a29a4ba-1957-4b4b-be8f-50fe2c6d8657.png)

2. Unity. Прикрепил скрипт на C# к камере

![image](https://user-images.githubusercontent.com/114305772/192608649-59c537b6-8233-4419-974e-d361e59a8647.png)

## Задание 2
### Ход работы:
1. Произвести подготовку данных для работы с алгоритмом линейной
регрессии. 10 видов данных были установлены случайным образом, и
данные находились в линейной зависимости. Данные преобразуются в
формат массива, чтобы их можно было вычислить напрямую при
использовании умножения и сложения.

```py

In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```
![image](https://user-images.githubusercontent.com/114305772/192612918-6f199c48-cbf3-4fdf-ad12-cd51d0620821.png)

2. Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.

```py

In [ ]:
#The basic linear regression model is wx+ b, and since this is a two-dimensional
space, the model is ax+ b
def model(a, b, x):
return a*x + b
#Tahe most commonly used loss function of linear regression model is the loss
function of mean variance difference
def loss_function(a, b, x, y):
num = len(x)
prediction=model(a,b,x)
return (0.5/num) * (np.square(prediction-y)).sum()
#The optimization function mainly USES partial derivatives to update two parameters a
and b
def optimize(a,b,x,y):
num = len(x)
prediction = model(a,b,x)
#Update the values of A and B by finding the partial derivatives of the loss
function on a and b
da = (1.0/num) * ((prediction -y)*x).sum()
db = (1.0/num) * ((prediction -y).sum())
a = a - Lr*da
b = b - Lr*db
return a, b
#iterated function, return a and b
def iterate(a,b,x,y,times):
for i in range(times):
a,b = optimize(a,b,x,y)
return a,b

```

![image](https://user-images.githubusercontent.com/114305772/192615390-95b8d1de-dc2a-4bd2-ab94-6c18da9d10c9.png)

3 Начать итерацию
Шаг 1. Инициализация и модель итеративной оптимизации.

```py

In [ ]:
#Initialize parameters and display
a = np.random.rand(1)
b = np.random.rand(1)
Lr = 0.000001
#For the first iteration, the parameter values, losses, and visualization after the
iteration are displayed
a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![image](https://user-images.githubusercontent.com/114305772/192617360-051852dc-8696-4226-8650-fb54b4fa12fa.png)

Шаг 2. На второй итерации отображаются значения параметров, значения
потерь и эффекты визуализации после итерации.

```py

In [ ]:
a,b = iterate(a,b,x,y,2)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![image](https://user-images.githubusercontent.com/114305772/192617781-7a725430-e256-4ee5-b76b-d213191b9478.png)

Шаг 3 Третья итерация показывает значения параметров, значения потерь и
визуализацию после итерации

```py

In [ ]:
a,b = iterate(a,b,x,y,3)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![image](https://user-images.githubusercontent.com/114305772/192617922-0596bf5e-5155-4033-b252-6449efe58814.png)

Шаг 4 На четвертой итерации отображаются значения параметров, значения
потерь и эффекты визуализации

```py

In [ ]:
a,b = iterate(a,b,x,y,4)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![image](https://user-images.githubusercontent.com/114305772/192618534-57ecf92e-50a8-48d5-9dfa-5b1b9b2d780a.png)

Шаг 5 Пятая итерация показывает значение параметра, значение потерь и
эффект визуализации после итерации

```py

In [ ]:
a,b = iterate(a,b,x,y,5)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![image](https://user-images.githubusercontent.com/114305772/192618771-c98d7959-26fa-4dac-9ac3-fedeeb1387eb.png)

Шаг 6 10000-я итерация, показывающая значения параметров, потери и
визуализацию после итерации

```py

In [ ]:
a,b = iterate(a,b,x,y,10000)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)

```

![image](https://user-images.githubusercontent.com/114305772/192618994-39d567b2-24ac-4949-800e-c71bc628e2df.png)

## Выводы

По окончании данной работы я научился использовать основные операторы языка Python на примере реализации линейной регрессии. При помощи оператора выдода сообщений в консоль, вывел текст "Hello World!" на Unity и Python.

| Plugin | README |
| ------ | ------ |
| GitHub | [plugins/github/README.md][PlGh] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
