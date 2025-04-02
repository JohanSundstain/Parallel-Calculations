import time
import queue
import threading
import argparse

import cv2
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#basic logger
logging.basicConfig(filename='log/logfile.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Sensor:
	def get(self):
		raise NotImplementedError("Subclass must implement method get()")
	
class SensorX(Sensor):
	'''SensorX'''
	def __init__(self, delay: float):
		self._delay = delay
		self._data = 0

	def get(self) -> int:
		time.sleep(self._delay)
		self._data += 1
		return self._data

class SensorCam(Sensor):
	'''SensorCam'''
	def __init__(self, camera, resolution):
		self._cap = cv2.VideoCapture(camera)
		self._width, self._height = resolution
		self._frames = queue.Queue(maxsize=1)
		self._run = True
		self._number_of_frame = 0

		if not self._cap.isOpened():
			logging.error("Camera not detected")
			raise ValueError("Cameara not detected")

		self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
		self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
		
		if self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) != self._width or self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != self._height:
			logging.error("Can't change resolution") 

		logging.info("Camera finded")
		
	def _read_cam(self):
		try:
			while self._run: 
				ret, frame = self._cap.read()
				self._number_of_frame += 1
				if not ret:
					logging.info("No frames. Checking the camer...")
					time.sleep(1)
					if not self._cap.isOpened():
						logging.info("Camera is turned off")
						self._run = False
						break
					else:
						continue

				if not self._frames.empty():
					self._frames.get()
				
				self._frames.put(frame)
			logging.info(f'{self._number_of_frame} read frames')
		except cv2.error:
			logging.error("Something wrong with camera")

	def get(self):
		if not self._run and self._frames.empty():  # Если поток завершился и кадров нет
			return None
		try:
			return self._frames.get(timeout=1)  # Ждём 1 секунду, если нет кадров
		except queue.Empty:
			return None
		
	def __del__(self):
		self._cap.release()
		logging.info("Camera is released")

	def stop(self):
		self._run = False

	def __call__(self):
		self._read_cam()

	def is_stoped(self):
		return self._run


class WindowImage:
    font_path = "arial.ttf"  
    base_font_size = 32  
    color = (255, 0, 0)
    
    def __init__(self, frequency):
        self._delay = int(1000 / frequency)
        self._window_name = "MainWindow"
        self._text = ""
        self._is_closed = False
        self.width = 1200
        self.height = 720
        
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, self.width, self.height)
        
        try:
            self.font = ImageFont.truetype(self.font_path, self.base_font_size)
        except IOError:
            logging.warning("Font not found, using default font")
            self.font = ImageFont.load_default()
    
    def show(self, image):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        scale_factor = min(self.width, self.height) / 800 
        font_size = int(self.base_font_size * scale_factor)
        font = ImageFont.truetype(self.font_path, font_size)
        position = (0, 0)
        
        draw.text(position, self._text, font=font, fill=self.color)
        image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        cv2.imshow(self._window_name, image_with_text)
        if cv2.waitKey(self._delay) & 0xFF == ord('q'):
            logging.info("The window is closed")
            self._is_closed = True
    
    def put_text(self, text):
        self._text = text
    
    def is_closed(self):
        return self._is_closed
    
    def __del__(self):
        logging.info("Window is destroyed")
        cv2.destroyAllWindows()


class SensorXTask():
	def __init__(self, sensor):
		self._sensor = sensor	
		self._data = 0
		self._run = True

	def __call__(self):
		while self._run:
			self._data = self._sensor.get()

	def get(self):
		return self._data

	def stop(self):
		self._run = False


def parse_args():
    parser = argparse.ArgumentParser(description="Настройки камеры")
    
    parser.add_argument("-c", "--camera", type=str, required=True)
    parser.add_argument("-r", "--resolution", type=str, default="720x600")
    parser.add_argument("-f", "--frequency", type=int, default=30)

    args = parser.parse_args()

    width, height = map(int, args.resolution.split("x"))
    
    return args.camera, (width, height), args.frequency

def main():
	cam_name, resolution, frequency = parse_args()

	with open("log/logfile.log", "w") as f:
		pass

	try:
		camera = SensorCam(int(cam_name), resolution)
		window = WindowImage(frequency)
	except ValueError:
		logging.error("Catch exception")
		exit()

	sensor0 = SensorX(0.01)
	sensor1 = SensorX(0.1)
	sensor2 = SensorX(1)

	task0 = SensorXTask(sensor0)
	task1 = SensorXTask(sensor1)
	task2 = SensorXTask(sensor2)

	thread0 = threading.Thread(target=camera)
	thread1 = threading.Thread(target=task0)
	thread2 = threading.Thread(target=task1)
	thread3 = threading.Thread(target=task2)
	thread0.start()
	thread1.start()
	thread2.start()
	thread3.start()

	while not window.is_closed():
		frame = camera.get()
		window.put_text(f"Sens1: [{task0.get()}]\nSens2: [{task1.get()}]\nSens3: [{task2.get()}]")
		if frame is not None and frame.size > 0:
			window.show(frame)
		else:
			break

	camera.stop()
	task0.stop()
	task1.stop()
	task2.stop()
	thread0.join()
	thread1.join()
	thread2.join()
	thread3.join()

if __name__ == "__main__":
	main()