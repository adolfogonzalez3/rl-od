import matplotlib.pyplot as plt
import numpy as np
import os

datas = np.array([])
y = []
averages = []
for filename in os.listdir("./"):
    if filename.endswith(".npy"):
        dat = np.load(filename)
        y.append(len(dat))
        averages.append(np.average(dat))
        
        datas = np.concatenate((datas, dat), axis=0)
        

datas = np.array(datas)


#plt.plot(averages[:5])
ranger = 0
ok = 0
x = []
yy = []
for _ in range(20):
    ranger += y[_] 
    ok = ranger - y[_]
    plt.axvline(x=ok, ymin=0, ymax=1, hold=None, color="black")
    yy.append(ok)

plt.plot(yy, averages[:20])
plt.show()


#plt.plot(averages[:5])
ranger = 0
ok = 0
x = []
yy = []
for _ in range(20):
    ranger += y[_ + 20] 
    ok = ranger - y[_ + 20]
    plt.axvline(x=ok, ymin=0, ymax=1, hold=None, color="black")
    yy.append(ok)

plt.plot(yy, averages[20:40])
plt.show()


#plt.plot(averages[:5])
ranger = 0
ok = 0
x = []
yy = []
for _ in range(20):
    ranger += y[_ + 40] 
    ok = ranger - y[_ + 40]
    plt.axvline(x=ok, ymin=0, ymax=1, hold=None, color="black")
    yy.append(ok)

plt.plot(yy, averages[40:60])
plt.show()


#plt.plot(averages[:5])
ranger = 0
ok = 0
x = []
yy = []
for _ in range(20):
    ranger += y[_ + 60] 
    ok = ranger - y[_ + 60]
    plt.axvline(x=ok, ymin=0, ymax=1, hold=None, color="black")
    yy.append(ok)

plt.plot(yy, averages[60:80])
plt.show()

#plt.plot(averages[:5])
ranger = 0
ok = 0
x = []
yy = []
for _ in range(20):
    ranger += y[_ + 80] 
    ok = ranger - y[_ + 80]
    plt.axvline(x=ok, ymin=0, ymax=1, hold=None, color="black")
    yy.append(ok)

plt.plot(yy, averages[80:100])
plt.show()

