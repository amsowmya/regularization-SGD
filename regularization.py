from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledataloader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="animals", help="Path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = simplepreprocessor.SimplePreprocessor(32, 32)
sdl = simpledataloader.SimpleDataLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

for r in (None, "l1", "l2"):
    # train a SGD classifier using a softmax loss function and the specified regularization
    # function for 10 epochs
    print(f"[INFO] training model with '{r}' penalty")
    model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant",
                          tol=1e-3, eta0=0.01, random_state=12)
    model.fit(trainX, trainY)

    # evaluate the classifiet
    acc = model.score(testX, testY)
    print(f"[INFO] '{r}' penalty accuracy: {acc * 100:.2f}%")
