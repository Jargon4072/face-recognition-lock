import cv2
import select
import sys
import config
import face
import hardware

def is_letter_input(letter):
	# Utility function to check if a specific character is available on stdin.
	# Comparison is case insensitive.
	if select.select([sys.stdin,],[],[],0.0)[0]:
		input_char = sys.stdin.read(1)
		return input_char.lower() == letter.lower()
	return False
if __name__ == '__main__':
	# Load training data into model
	print( 'Loading training data...')
	model = cv2.face.createEigenFaceRecognizer()
	model.load(config.TRAINING_FILE)
	print( 'Training data loaded!')
	# Initialize camer and box.
	camera = config.get_camera()
	box = hardware.Box()
	# Move box to locked position.
	box.lock()
	print( 'Running box...')
	print ('Press button to lock (if unlocked), or unlock if the correct face is detected.')
	print( 'Press Ctrl-C to quit.')

	while True:
		# Check if capture should be made.
		# TODO: Check if button is pressed.
		if box.is_button_up() or  is_letter_input('c'):
			if not box.is_locked:
				# Lock the box if it is unlocked
				box.lock()
				print ('Box is now locked.')
			else:
				print ('Button pressed, looking for face...')
				# Check for the positive face and unlock if found.
				image = camera.read()
				# Convert image to grayscale.
				image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				# Get coordinates of single face in captured image.
				result = face.detect_single(image)
				if result is None:
					print( 'Could not detect single face!  Check the image in capture.pgm' \
						  ' to see what was captured and try again with only one face visible.')
					continue
				x, y, w, h = result
				# Crop and resize image to face.
				crop = face.resize(face.crop(image, x, y, w, h))
				# Test face against model.
				label, confidence = model.predict(crop)
				#result = cv2.face.MinDistancePredictCollector()
				#model.predict(crop)
				#confidence = result.getDist()
				#label=result.getLabel()
				print( 'Predicted {0} face with confidence {1} (lower is more confident).'.format(
					'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE',
					confidence))
				if label == config.POSITIVE_LABEL and (confidence < config.POSITIVE_THRESHOLD and confidence >1500):
					print ('Recognized face!')
					box.unlock()
				else:
					print( 'Did not recognize face!')
