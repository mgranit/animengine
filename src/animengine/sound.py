import numpy as np
from scipy.io.wavfile import write
from scipy.signal import resample
import soundfile as sf

class SoundMixer:
    def __init__(self, duration, sample_rate=44100):
        
        self.duration = duration
        self.sample_rate = sample_rate
        self.recording_buffer = np.zeros(int(duration * sample_rate))
        self.buffer_len = len(self.recording_buffer)
        
    def add_to_recording(self, audio_chunk, current_time):
        
        chunk_length = len(audio_chunk)
        start_index = int(current_time * self.sample_rate)
        end_index = start_index + chunk_length

        # Clip indices to fit within the buffer
        if start_index >= self.buffer_len:
            return  # Entire chunk is out of range
        if end_index > self.buffer_len:
            audio_chunk = audio_chunk[:self.buffer_len - start_index]
            end_index = self.buffer_len

        self.recording_buffer[start_index:end_index] += audio_chunk
    
    def process(self, method='normalize', save=False, filename='sound.wav'):
        if method == 'normalize':
            max_val = np.max(np.abs(self.recording_buffer))
            if max_val > 0:
                self.recording_buffer /= max_val
        elif method == 'soft_clip':
            self.recording_buffer = np.tanh(self.recording_buffer)
        
        if save:
            write(filename, self.sample_rate, (self.recording_buffer * 32767).astype(np.int16))

def create_bounce_soundwave(
    duration=1.0,
    frequency=440.0,
    amplitude=1.0,
    attack_duration=0.001,
    release_tau=0.1,
    sample_rate=44100
    ):

    num_samples = int(sample_rate * duration)
    time_array = np.linspace(0.0, duration, num_samples, endpoint=False)
    samples = np.sin(2 * np.pi * frequency * time_array)
    samples += 0.5 * np.sin(2 * np.pi * np.random.uniform(1.25, 1.75) * frequency * time_array)
    samples += 0.25 * np.sin(2 * np.pi * np.random.uniform(2.25, 2.75) * frequency * time_array)

    # Attack
    num_attack_samples = int(sample_rate * attack_duration)
    attack_time_array = time_array[:num_attack_samples]
    attack_envelope = np.exp((np.log(2) / attack_duration) * attack_time_array) - 1
    samples[:num_attack_samples] *= attack_envelope

    # Release
    release_time_array = time_array[num_attack_samples:]
    release_time_start = release_time_array[0]
    release_time_end = release_time_array[-1]
    release_envelope = (np.exp(-(release_time_array - release_time_start) / release_tau) - np.exp(-(release_time_end - release_time_start) / release_tau)) / (1 - np.exp(-(release_time_end - release_time_start) / release_tau))
    samples[num_attack_samples:] *= release_envelope

    samples *= amplitude
    
    samples /= np.max(np.abs(samples))

    assert samples[0] == 0.0
    assert samples[-1] == 0.0
    assert np.max(np.abs(samples)) <= 1.0
    
    return samples

def load_audio_file(filepath, target_sr=44100):
    data, sr = sf.read(filepath)
    
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    data = data.astype(np.float64)
    
    if sr != target_sr:
        new_len = int(len(data) * target_sr / sr)
        data = resample(data, new_len)
        
    fade = int(0.1 * target_sr)
    if fade > 0 and len(data) > 2*fade:
        win_in = np.linspace(0.0, 1.0, fade)
        win_out = np.linspace(1.0, 0.0, fade)
        data[:fade] *= win_in
        data[-fade:] *= win_out
    
    return data
