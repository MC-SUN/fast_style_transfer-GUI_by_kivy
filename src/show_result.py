import cv2
def show_video(path):
	#path = '../' + path
	cap=cv2.VideoCapture(path)
	while(cap.isOpened()):
		ret,frame=cap.read()
		cv2.putText(frame, "press Q to go back", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (200, 200, 200), 5)
		cv2.imshow(path,frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
	return True
def show_image(path):
	#path='../'+path
	image=cv2.imread(path)
	cv2.putText(image,"press Q to go back", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (200, 200, 200), 5)
	while True:
		cv2.imshow("Image After Style Transfer",image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cv2.destroyAllWindows()
	return True

if __name__=='__main__':
	#print(show_video("video/1.mp4"))

	print(show_image('image/content1.jpg'))
