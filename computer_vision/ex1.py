import cv2
print(cv2.__version__)
img=cv2.imread("pomme.jpg",1)
print(img)
cv2.imshow("first image",img)
k=cv2.waitKey(0)

if k==27:
    cv2.destroyAllWindows()
elif k==ord("s"):
    cv2.imwrite("second image",img)
    cv2.destroyAllWindows() 

