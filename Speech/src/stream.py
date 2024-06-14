import pyaudio as pa
import time
import numpy as np


class Streamer:
    def __init__(self, chunk_size, func):
        self.pa = pa.PyAudio()
        self.sr = 16000
        self.chunk_size = chunk_size
        self.func = func
        self.device_index = self._setup_input_device()
        self.buffer_size = int(self.sr * self.chunk_size / 1000)
    
    def _setup_input_device(self):
        print('Available audio input devices:')
        input_devices = []
        for i in range(self.pa.get_device_count()):
            dev = self.pa.get_device_info_by_index(i)
            if dev.get('maxInputChannels'):
                input_devices.append(i)
                print(i, dev.get('name'))

        if len(input_devices):
            dev_idx = -2
            while dev_idx not in input_devices:
                print('Please type input device ID:')
                dev_idx = int(input())
        else:
            raise RuntimeError('No audio input device found.')
        return dev_idx


    def _callback(self, in_data, frame_count, time_info, status):
        signal = np.frombuffer(in_data, dtype=np.int16)
        res = self.func(signal)
        print(res, end='\r')
        return (in_data, pa.paContinue)

    def start(self):
        stream = self.pa.open(
            format=pa.paInt16,
            channels=1,
            rate=self.sr,
            input=True,
            input_device_index=self.device_index,
            stream_callback=self._callback,
            frames_per_buffer=self.buffer_size - 1
        )
        print('Listening...')
        stream.start_stream()

        # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
        try:
            while stream.is_active():
                time.sleep(0.1)
        finally:        
            stream.stop_stream()
            stream.close()
            self.pa.terminate()
            print("Stopped listening.")
