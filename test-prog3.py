import numpy as np
import matplotlib.pyplot as plt


plt.rc('font', size=16)


t = np.linspace(0, 1000, 1001)
p0 = 200
t0 = 1000

# Смоделируем измерения с погрешностью
p = p0 / (1 + t / t0) + 0.1 * np.random.randn(len(t))

# Истинное значение производной - для сравнения
pprime = -p0 / t0 / (1 + t / t0)**2; 


def diff1(t, p):
    # Функция np.diff возвращает массив соседних разностей - как раз то, что нужно
    return np.diff(p) / np.diff(t)

def diff2(t, p):
    # Разность через одну точку удобнее считать с помощью операций с частями массива 
    return (p[2:] - p[:-2]) / (t[2:] - t[:-2])

def diffn(t, p, n):
    return ((p[n:] - p[:-n]) / (t[n:] - t[:-n]))

def diff20(t, p):
    return (p[20:] - p[:-20]) / (t[20:] - t[:-20])


# plt.figure(figsize=(16, 5))
# plt.subplot(1, 2, 1)
# plt.plot(t[1:-1], diff2(t, p), 'g.', label="diff2")
# plt.plot(t, pprime, 'r-', label="p'(t)")
# plt.title('error = ' + str(np.abs(diff2(t, p) - pprime[1:-1]).max()))
# plt.legend(loc='best')
# plt.subplot(1, 2, 2)
# plt.plot(t[10:-10], diff20(t, p), 'm.', label="diff20")
# plt.plot(t, pprime, 'r-', label="p'(t)")
# plt.title('error = ' + str(np.abs(diff20(t, p) - pprime[10:-10]).max()))
# plt.legend(loc='best')
# for i in range(1, 100):
#     plt.figure(figsize=(16, 5))
#     plt.subplot(1, 2, 2)
#     plt.plot(t[i:-i], diffn(t, p, 2 * i), 'm.', label=f"diff{2 * i}")
#     plt.plot(t, pprime, 'r-', label="p'(t)")
#     plt.title('error = ' + str(np.abs(diffn(t, p, 2 * i) - pprime[i:-i]).max()))
#     plt.legend(loc='best')









# Подогнать данные (t_i, p_i) многочленом пятой степени
# full=True дает доступ к расширенной информации - ошибке приближения
# *_ - отбрасывает все лишние результаты, после первых двух
coeff, [err], *_ =  np.polyfit(t, p, 5, full=True)


plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
plt.plot(t, p, label='$p(t_i)$')
plt.plot(t, np.polyval(coeff, t), 'r-', label='$P_5(t)$')
plt.xlabel('t'); plt.ylabel('p'); plt.legend(loc='best');
plt.subplot(1, 2, 2)
plt.plot(t[400:500], p[400:500], label='$p(t_i)$')
plt.plot(t[400:500], np.polyval(coeff, t[400:500]), 'r-', label='$P_5(t)$')
plt.xlabel('t'); plt.ylabel('p'); plt.legend(loc='best');


degs = list(range(0, 15))
errs = []
for deg in degs:
    # Сейчас нас интересует только ошибка приближения
    _, [err], *_ =  np.polyfit(t, p, deg, full=True)
    errs.append(err)
    
plt.semilogy(degs, errs, '.')
plt.annotate("Optimum", xy=(degs[4], errs[4]), xytext=(3.5, 1e3), 
             arrowprops={"arrowstyle":"-|>"})
plt.xlabel('$n$')
plt.ylabel('$\sum r_i^2$')