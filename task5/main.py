import argparse
import time

import threading
import queue
from ultralytics import YOLO
import cv2

stop_event = threading.Event()

finished = queue.Queue(100) 
raw = queue.Queue(100)

class Worker:
	def __init__(self):
		self._yolo = YOLO("yolov8n-pose.pt", verbose=False).to("cpu")
	
	def __call__(self):
		while not stop_event.is_set():
			try:
				index, frame = raw.get(timeout=0.5)
				result = self._yolo(frame, verbose=False)
				finished.put((index, result[0].plot()), block=True) 

			except queue.Empty:
				continue

	
class Reader:
	def __init__(self, video_path):
		self._global_index = 0
		self._cap = cv2.VideoCapture(video_path)

	def __call__(self):
		while not stop_event.is_set():
			ret, frame = self._cap.read()
			if not ret:
				stop_event.set()
				break
			raw.put((self._global_index, frame), block=True)
			self._global_index += 1
	

class Display:
	def __init__(self):
		self._window_name = "win"
		cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(self._window_name, width=640, height=480)
		self._showed_index = 0
		self._buffer = {}

	def __call__(self):
		while not stop_event.is_set():
			try:
				index, frame = finished.get(timeout=1)
				self._buffer[index] = frame

				while self._showed_index in self._buffer:
					current_frame = self._buffer.pop(self._showed_index)
					self._showed_index += 1

					cv2.imshow(self._window_name, current_frame)
					cv2.waitKey(30)

			except queue.Empty:
				continue
			

	def __del__(self):
		cv2.destroyAllWindows()


def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 Pose Detection")
	parser.add_argument('-v', "--video",type=str, default="video.mp4", help="Path to the video file")
	parser.add_argument('-w',"--workers", type=int, default=2, help="Number of worker threads")
	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	video_path = args.video
	workers = args.workers

	reader = Reader(video_path)
	theads = []
	start = time.time()
	for i in range(workers):
		worker = Worker()
		theads.append(threading.Thread(target=worker))
		theads[i].start()

	reader_thread = threading.Thread(target=reader)
	reader_thread.start()

	display = Display()
	display()

	for i in range(workers):
		theads[i].join()
	
	reader_thread.join()
	end = time.time() - start
	print(f"Time: {end:.2f} sec")


if __name__  == "__main__":
	main()