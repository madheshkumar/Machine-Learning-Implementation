import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data1.csv")

# linear regression for single variable : y = mx +c
def gradientDescent(m_cur, c_cur, points, alpha):
    m_gradient = 0
    c_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].annual
        y = points.iloc[i].start

        m_gradient += -(2 / n) * x * (y - (m_cur * x + c_cur))
        c_gradient += -(2 / n) * (y - (m_cur * x + c_cur))

    m = m_cur - m_gradient * alpha
    c = c_cur - c_gradient * alpha
    return m, c

m = 0
c = 0
alpha = 0.001
iters = 800

for i in range(iters):
    m, c = gradientDescent(m, c, data, alpha)

print(m, c)
plt.scatter(data.annual, data.start, color="black")
plt.plot(list(range(0, 4)), [m * x + c for x in range(0,4)], color="blue")
plt.show()
