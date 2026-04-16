import matplotlib.pyplot as plt
import os, psutil
for i in range(3000):
    fig = plt.figure()
    plt.plot([1, 2], [3, 4])
    plt.close(fig)
print(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
