from ledcd import CubeDrawer as cd

from playsound import playsound
import threading

import matplotlib.pyplot as plt

from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile

import numpy as np
import time
import math

plt.ion()

file_name = (
    "C://Users//user//Desktop//3D-Led-Cube//examples//audio_visualizer//song.wav"
)

rate, data = wavfile.read(file_name)

if len(data.shape) != 2:
    (shape_size,) = data.shape
    data = np.concatenate([data, data], axis=None).reshape((shape_size, 2))

start_frame = 0
frame_size = rate // 2
half_frame = frame_size // 2

fig = plt.figure()

ax = fig.add_subplot(111)

yf = rfft(data[start_frame : start_frame + frame_size, 0])
xf = np.log(rfftfreq(frame_size, 1 / rate) + 25)
(g,) = ax.plot(xf, np.abs(yf))  # Returns a tuple of line objects, thus the comma
(h,) = ax.plot(xf, np.abs(yf))  # Returns a tuple of line objects, thus the comma

ax.set_ylim([0, 100000000])
prev_time = time.time()
frames_min_delay = 1 / rate * frame_size

threading.Thread(target=lambda: playsound(file_name), daemon=True).start()

start_time = time.time()
while True:
    yfl = rfft(data[start_frame : start_frame + frame_size, 0])
    yfr = rfft(data[start_frame : start_frame + frame_size, 1])

    g.set_ydata(np.abs(yfl))
    h.set_ydata(np.abs(yfr))

    fig.canvas.draw()
    fig.canvas.flush_events()

    start_frame = int((time.time() - start_time) * rate)

    time.sleep(1 / 60)
