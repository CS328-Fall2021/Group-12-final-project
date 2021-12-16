import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from time import sleep

def record():
    print('Recording in 3')
    sleep(1)
    print('Recording in 2')
    sleep(1)
    print('Recording in 1')
    sleep(1)
    print('>>>>>>>>>>>>>>>>>>> Start Recording<<<<<<<<<<<<<<<<<<<')
    frequency = 44400
    # freq = frequency

    duration = 5
    recording = sd.rec(int(duration * frequency),samplerate = frequency, channels = 2)
    sd.wait()
    write("audio_file.wav", frequency, recording)
