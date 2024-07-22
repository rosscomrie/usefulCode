### uses local (to host machine) audio - not suitable for non-browser applications

import keyboard
import threading
import wave
import pyaudio
import os

class AudioRecorder:
    def __init__(self, start_key='q', stop_key='w', directory='tmp/audio', file_name='audio.wav'):
        self.directory = directory if directory.endswith("\\") else directory + "\\"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.audio_path = self.directory + file_name
        self.file_name = file_name
        self.start_key = start_key
        self.stop_key = stop_key
        self.recording = False
        self.frames = []
        self.THREAD = threading.Thread()
 
    def start(self, e):
        if not self.recording:
            print('Starting recording...')
            self.recording = True
            self.frames.clear()
            self.THREAD = threading.Thread(target=self.capture_audio)
            self.THREAD.start()

    def stop(self):
        if self.recording:
            print('Stopping recording...')
            self.recording = False
            self.THREAD.join()

    def capture_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
 
        p = pyaudio.PyAudio()

        # start Recording
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        print("Recording...")
        while self.recording:
            data = stream.read(CHUNK)
            self.frames.append(data)
        print("Finished recording.")

        # stop Recording
        stream.stop_stream()
        stream.close()
        p.terminate()

        print("Saving audio...")
        wf = wave.open(self.directory + self.file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print("Finished saving audio to " + self.directory + self.file_name)