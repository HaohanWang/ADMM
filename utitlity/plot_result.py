
from matplotlib import pyplot as plt
import numpy as np

def parse_text(filename, cleaned=False):
    r = []
    time = 0.0
    t = 0.0
    v = 0.0
    e = 0.0
    first = True
    text = [line.strip() for line in open(filename)]
    if cleaned:
        for line in text:
            if line.find('updating time usage')!=-1:
                if first:
                    first = False
                else:
                    r.append((t, v, e))
                t = float(line.split(':')[-1]) + time
                time = t
            if line.find('validation error')!=-1:
                v = 1 - float(line.split(':')[-1])*0.01
            if line.find('test error')!=-1:
                e = 1 - float(line.split(':')[-1])*0.01
        r.append((t, v, e))
    else:
        for line in text:
            if line.find('updating time usage')!=-1:
                if first:
                    first = False
                else:
                    r.append((t, v, e))
                t = float(line.split()[-1]) + time
                time = t
            if line.find('validation error')!=-1:
                v = 1 - float(line.split()[-2])*0.01
            if line.find('test error')!=-1:
                e = 1 - float(line.split()[-2])*0.01
        r.append((t, v, e))
    return r

r1 = parse_text('../deep_learning/result_original_mlp.txt', cleaned=False)
r2 = parse_text('../deep_learning/result_6000.txt', cleaned=False)

t1, v1, e1 = zip(*r1)
t2, v2, e2 = zip(*r2)

c1 = np.arange(len(t1))
c2 = np.arange(len(t2))

plt.plot(t1, v1, c='g', label='gd, validation')
plt.plot(t1, e1, c='b', label='gd, test')

plt.plot(t2, v2, c='r', label='admm, validation')
plt.plot(t2, e2, c='m', label='admm, test')
plt.xlabel('time elapses')
plt.ylabel('accuracy')
plt.legend(loc=4)

plt.show()

plt.plot(c1, v1, c='g', label='gd, validation')
plt.plot(c1, e1, c='b', label='gd, test')

plt.plot(c2, v2, c='r', label='admm, validation')
plt.plot(c2, e2, c='m', label='admm, test')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.legend(loc=4)

plt.show()