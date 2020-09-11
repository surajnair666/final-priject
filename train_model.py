from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# training
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# writeclabel encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()