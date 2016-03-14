from myro import *
import math

init("/dev/tty.Fluke2-0B3B-Fluke2")

def takePics():
    turnBy(5, "deg")
    print "1"
    pic1 = takePicture()
    show(pic1)
    print "here"
    savePicture(pic1, "pic1.jpg")

    turnBy(10, "deg")
    print "2"
    pic2 = takePicture()    
    savePicture(pic2, "pic2.jpg")

    turnBy(15, "deg")
    print "3"
    pic3 = takePicture()    
    savePicture(pic3, "pic3.jpg")
    
    turnBy(20, "deg")
    print "4"
    pic4 = takePicture()
    savePicture(pic4, "pic4.jpg")

    turnBy(25, "deg")
    print "5"
    pic5 = takePicture()
    savePicture(pic5, "pic5.jpg")

    turnBy(30, "deg")
    pic6 = takePicture()
    savePicture(pic6, "pic6.jpg")

    turnBy(35, "deg")        
    pic7 = takePicture()
    savePicture(pic7, "pic7.jpg")

    turnBy(40, "deg")
    pic8 = takePicture()
    savePicture(pic8, "pic8.jpg")

    turnBy(45, "deg")
    pic9 = takePicture()
    savePicture(pic9, "pic9.jpg")
    print "got here bish"

takePics()
