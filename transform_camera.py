import datetime
import cv2
import evaluate
def camera_transfer(checkpoint_dir,screen_shot_dir="output/camera_screen/"):
    cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    fps=cap.get(30)
    i=0
    date=[]
    date2=datetime.datetime.now()
    while(True):
        ret,frame=cap.read()
        i=i+1
        if i/fps==int(i/fps):
            cv2.imwrite("Assets/image/tmp.png",frame)
            evaluate.ffwd_to_img("Assets/image/tmp.png","Assets/image/tmp2.png",checkpoint_dir)
            frame=cv2.imread("Assets/image/tmp2.png")
            cv2.imshow('frame',frame)
            date1=datetime.datetime.now()
            date.append(date1 -date2)
            date2=date1
        if cv2.waitKey(1)&0xFF==ord('s'):
            s=screen_shot_dir+"Screen_shot"+str(datetime.datetime.now()).split(" ")[0]+"-"+(str(datetime.datetime.now()).split(" ")[1].split(".")[0]).replace(":","-")+'.png'
            cv2.imshow("screen shot",frame)
            cv2.imwrite(s,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("你按了Q键")
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__=='__main__':
    camera_transfer("style/scream.ckpt")