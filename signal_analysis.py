import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.io import wavfile

# 1. WAV-Datei einlesen
sample_rate, signal = wavfile.read('sig2_2024_africa.wav')

# 2. Falls das Signal Stereo ist, auf Mono reduzieren
if len(signal.shape) == 2:
    signal = signal.mean(axis=1)

# 3. Signalparameter festlegen
n = len(signal)
duration = n / sample_rate
t = np.linspace(0.0, duration, n, endpoint=False)

# 4. FFT des Signals berechnen
yf = fft(signal)
xf = fftfreq(n, 1 / sample_rate)

# 5. Nur positive Frequenzen verwenden
positive_freqs = xf[:n // 2]
magnitude = 2.0 / n * np.abs(yf[:n // 2])

plt.ion()

# 6. Frequenzspektrum plotten
plt.figure(figsize=(10, 6))
plt.plot(positive_freqs, magnitude)
plt.title('Frequenzspektrum der WAV-Datei')
plt.xlabel('Frequenz (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# 7. Optional: Originalsignal im Zeitbereich plotten
plt.figure(figsize=(10, 6))
plt.plot(t, signal)
plt.title('Zeitbereich des Signals')
plt.xlabel('Zeit (Sekunden)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
