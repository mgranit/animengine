import pygame
import datetime
import os
from PIL import Image
import inspect
import numpy as np

import OpenGL.GL as GL
from .openglutils import Shader, Uniform, TextureData, read_shader, compile_shader, initialize_program
from .sound import SoundMixer
from .usefuls import gaussian
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

class Simulation:
    def __init__(
        self,
        shader,
        resolution,
        duration,
        fps,
        spf=1,
        reversal=False,
        reversal_time=0.0,
        slowdown_duration=0.0,
        record=False
    ):
        self.shader = shader
        self.width, self.height = resolution
        
        # frames
        self.fps = fps
        self.n_frames = int(duration * self.fps)
        self.frame = 0
        self.frame_str_width = len(str(self.n_frames))
        self.frame_str = f"{self.frame:0{self.frame_str_width}d}"
        
        # time
        self.duration = duration
        self.time_vid = 0.0
        self.time_sim = 0.0
        
        # simulation steps
        self.spf = spf
        self.step = 0
        self.dt_vid = 1 / (self.fps * self.spf)
        self.dt_sim = self.dt_vid
        
        # Reversal parameters
        self.reversal = reversal
        self.reversal_time = reversal_time
        self.slowdown_duration = slowdown_duration
        self.reversed = False
        
        # set up pygame and OpenGL
        pygame.init()
        self.clock = pygame.time.Clock()
        self.window = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
        self.shader.setup()
        
        # Recording
        self.record = record
        if self.record:
            
            # Determine caller directory
            # Find first stack frame outside this module so caller_dir points to the caller script
            caller_file = None
            for frame_info in inspect.stack()[1:]:
                fname = os.path.abspath(frame_info.filename)
                if fname != os.path.abspath(__file__):
                    caller_file = fname
                    break
            if caller_file is None:
                caller_file = os.path.abspath(inspect.stack()[1].filename)
            self.caller_dir = os.path.dirname(caller_file)
            
            self.timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            self.recordingsfolder = os.path.join(self.caller_dir, 'recordings', self.timestamp)
            os.makedirs(self.recordingsfolder, exist_ok=True)
            os.makedirs(os.path.join(self.recordingsfolder, 'frames'), exist_ok=True)
            params_path = os.path.join(self.caller_dir, 'params.txt')
            if os.path.exists(params_path):
                shutil.copy(params_path, self.recordingsfolder)
        
    def run(self):
        
        self.running = True
        while self.running:
            self.handle_events()
            self.update_func()
            
            if self.reversal:
                dt_scaling = (1.0 - gaussian(self.time_vid, 1.0, self.reversal_time, self.slowdown_duration / np.sqrt(2.0 * np.log(2.0))))
                self.dt_sim = self.dt_vid * dt_scaling
                
                if (not self.reversed) and (self.time_vid >= self.reversal_time):
                    self.reversed = True
                    self.reversal_func()
            
            if self.frame == self.n_frames - 1:
                self.running = False
            
            if self.step % self.spf == 0:
                self.update_frame()
            
            self.time_vid += self.dt_vid
            self.time_sim += self.dt_sim
            self.step += 1
            
            if self.step % self.spf == 0:
                self.frame += 1
                self.frame_str = f"{self.frame:0{self.frame_str_width}d}"
        
        self.shutdown()
    
    def handle_events(self):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and not self.record:
                    self.frame = 0
                    self.time = 0.0
                    self.step = 0
                    self.reset_func()
                        
    def update_frame(self):
        
        pygame.display.set_caption(f"Time: {self.time_vid:.2f}, Frame: {self.frame_str}")
        
        self.shader.draw(frame_num=self.frame)
        
        if self.record:
            pixels = GL.glReadPixels(0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            pixels_bytes = bytes(pixels)
            image = Image.frombytes("RGB", (self.width, self.height), pixels_bytes)
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image.save(os.path.join(self.recordingsfolder, 'frames', f"frame_{self.frame_str}.png"))
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def shutdown(self):
        pygame.quit()
        
        if self.record:
            os.system(f"ffmpeg -framerate 60 -i {self.recordingsfolder}/frames/frame_%0{self.frame_str_width}d.png -c:v libx264 -preset slower -crf 17 -pix_fmt yuv420p -profile:v high -level 5.1 {self.recordingsfolder}/video_{self.timestamp}.mp4")
            shutil.rmtree(os.path.join(self.recordingsfolder, 'frames'))
    
    def update_func(self):
        pass
    
    def reset_func(self):
        pass
    
    def reversal_func(self):
        pass

class SoundSimulation(Simulation):
    def __init__(
        self,
        shader,
        resolution,
        duration,
        fps,
        spf=1,
        reversal=False,
        reversal_time=0.0,
        slowdown_duration=0.0,
        record=False,
        sample_rate=44100
    ):
        super().__init__(shader, resolution, duration, fps, spf, reversal, reversal_time, slowdown_duration, record)
        
        # Sound mixer
        self.sample_rate = sample_rate
        self.sound_mixer = SoundMixer(duration=duration, sample_rate=self.sample_rate)
    
    def run(self):
        
        pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
        
        # Develop sound
        self.running = True
        while self.running:
            
            if self.reversal:
                dt_scaling = (1.0 - gaussian(self.time_vid, 1.0, self.reversal_time, self.slowdown_duration / np.sqrt(2.0 * np.log(2.0))))
                self.dt_sim = self.dt_vid * dt_scaling
                
                if (not self.reversed) and (self.time_vid >= self.reversal_time):
                    self.reversed = True
                    self.reversal_func()
            
            if self.frame == self.n_frames - 1:
                self.running = False
            
            self.update_func()
            self.time_vid += self.dt_vid
            self.time_sim += self.dt_sim
            self.step += 1
            self.frame += 1
        
        self.sound_mixer.process(method='normalize', save=self.record, filename=os.path.join(self.recordingsfolder, 'sound.wav') if self.record else 'sound.wav')
        
        self.frame = 0
        self.time_vid = 0.0
        self.time_sim = 0.0
        self.step = 0
        self.reversed = False
        self.reset_func()
        
        # Run visual simulation
        if self.sound_mixer.recording_buffer is not None:
            samples_int16 = (self.sound_mixer.recording_buffer * 32767).astype(np.int16)
            samples_int16 = np.column_stack((samples_int16, samples_int16))
            sound = pygame.sndarray.make_sound(samples_int16)
            sound.play()

        super().run()
    
    def shutdown(self):
        pygame.quit()
        
        if self.record:
            frames_pattern = os.path.join(self.recordingsfolder, 'frames', f"frame_%0{self.frame_str_width}d.png")
            video_path = os.path.join(self.recordingsfolder, f"video_{self.timestamp}.mp4")
            sound_path = os.path.join(self.recordingsfolder, 'sound.wav')
            if os.path.exists(sound_path):
                cmd = (
                    f"ffmpeg -y -framerate 60 -i \"{frames_pattern}\" -i \"{sound_path}\" "
                    f"-map 0:v:0 -map 1:a:0 -c:v libx264 -preset slower -crf 17 -pix_fmt yuv420p -profile:v high -level 5.1 "
                    f"-c:a aac -b:a 192k -shortest -movflags +faststart \"{video_path}\"")
            else:
                cmd = (
                    f"ffmpeg -framerate 60 -i \"{frames_pattern}\" -c:v libx264 -preset slower -crf 17 -pix_fmt yuv420p -profile:v high -level 5.1 \"{video_path}\"")
            os.system(cmd)
            shutil.rmtree(os.path.join(self.recordingsfolder, 'frames'))
