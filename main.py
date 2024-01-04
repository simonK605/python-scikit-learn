import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

# This is the image of the 7
print(digits.images[8])

# This is the digits data
print(len(digits.data))

clf = svm.SVC(gamma=0.001, C=100)
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)

print('Prediction: ', clf.predict(digits.data[-1].reshape(1, -1)))
# Show the image
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()