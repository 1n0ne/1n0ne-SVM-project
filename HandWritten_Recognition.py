'''
handwritten digit recognition using svm algorithm
'''
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from skimage.io import imread
from skimage.exposure import rescale_intensity
from skimage.transform import resize

#load the data
digits = load_digits()
#the data have 2 form in as data and as imges
#to see the number from data we have to reshape it in 8x8 in form of array

print(digits['data'][1].reshape(8,8))
#but to see it from images we only have to print it
(digits['images'][0])
#the model predict or give an output as target to see them write:
(digits['target'][0:10])
#split the data
mainData=digits['data']
targets=digits['target']
#train
svc=svm.SVC(gamma=0.001, C=100)
svc.fit(mainData[:1500], targets[:1500])
predictions=svc.predict(mainData[1501:])
#show predictions result
(list(zip(predictions,targets[1501:])))

#Calculating Confusion Matrix
CM = confusion_matrix(predictions,targets[1501:])
'''
# drawing confusion matrix
'''
sns.heatmap(CM, annot=True,cmap="YlGnBu")
plt.show()
'''
#show the accuracy report
'''
print(classification_report(predictions,targets[1501:]))
#saving model
joblib.dump(svc, "svmRmodel") 
#load it as model
#chick our work
model=joblib.load('svmRmodel')
img = resize(imread(('testnumber/7.jpg')), (8,8))

img = rescale_intensity(img, out_range=(0, 16))

x_test = [sum(pixel)/3.0 for row in img for pixel in row]

print("The predicted digit is {}".format(model.predict([x_test])))

print(digits.data.shape)
plt.gray()
plt.matshow(digits.images[2])


