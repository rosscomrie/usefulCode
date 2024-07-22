import io
import requests
import base64
from typing import Any
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from simpleaudio import WaveObject
import base64
import requests
import pyaudio
import io
import os
import threading
from pydub import AudioSegment
import random

class TextToSpeech:
    """
    This class uses the Eleven Labs API to convert text to speech.
    """

    def __init__(self, voice_id: str, elevenlabs_key: str, model_id: str, optimization_level: int = 0):
        """
        Initialize TextToSpeech class.
        :param voice_id: Voice ID for the desired voice output.
        :param elevenlabs_key: API key for Eleven Labs.
        :param optimization_level: Latency optimization level.
        """
        self.voice_id = voice_id
        self.model_id = model_id
        self.elevenlabs_key = elevenlabs_key
        self.optimization_level = optimization_level
        self.CHUNK_SIZE = 1024
        self.url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?optimize_streaming_latency={self.optimization_level}"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_key
        }

    def convert_to_audio_and_save(self, text, directory, filename):
        """
        Convert the text to speech and save it as a .wav file.
        """
        audio_bytes = self.convert(text)
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        if not os.path.exists(directory):
            os.makedirs(directory)  # If the directory doesn't exist, create it
        audio_seg.export(os.path.join(directory, filename+".wav"), format="wav")
        
    def generate_filler_sounds(self, filler_texts):
        """
        Generate filler sounds
        """
        directory = "/tmp/audio/utterances"
        for text in filler_texts:
            self.convert_to_audio_and_save(text, directory, text)
            
    def play_filler_sound(self, filler_sounds):
        """
        Randomly select a filler sound and play it.
        """
        directory = "/tmp/audio/utterances"
        selected_sound = random.choice(filler_sounds)
        wave_obj = WaveObject.from_wave_file(os.path.join(directory, selected_sound+'.wav'))
        play_obj = wave_obj.play()
        play_obj.wait_done()

    def convert(self, prompt: str, filler_sounds=None) -> bytes:
        """
        Converts text prompt to speech (audio).
        :param prompt: Text prompt to be converted to speech.
        :param filler_sounds: (optional) list of filler sounds to be played.
        :return: The audio bytes corresponding to the input text.
        """
        data = {
            "text": prompt,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0,
                "similarity_boost": 0
            }
        }

        # If filler_sounds are provided, play a random one 
        if filler_sounds is not None:
            threading.Thread(target=self.play_filler_sound, args=(filler_sounds,), daemon=True).start()

        response = requests.post(self.url, json=data, headers=self.headers, stream=True, verify=False)
        
        # If filler_sounds are provided, play a random one 
        if filler_sounds is not None:
            threading.Thread(target=self.play_filler_sound, args=(filler_sounds,), daemon=True).start()

        audio_bytes = b''
        for chunk in response.iter_content(chunk_size=self.CHUNK_SIZE):
            if chunk:
                audio_bytes += chunk
        return audio_bytes
    

    def audio_bytes_to_base64(audio_bytes):
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return audio_base64
    

    def play_audio(self, audio_bytes):
        """
        Play audio bytes.
        :param audio_bytes: Audio data in bytes.
        """
        # Convert bytes to .wav AudioSegment
        audio_seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

        # Convert AudioSegment to wav
        byte_io = io.BytesIO()
        audio_seg.export(byte_io, format="wav")
        byte_io.seek(0)

        # Load and play wav audio
        wave_obj = WaveObject.from_wave_file(byte_io)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until sound has finished playing