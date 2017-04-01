# Arda Mavi
from sklearn.neural_network import MLPClassifier

def egitim(X,y):
    # X => Eğitim için örnek giriş verileri.
    # y => Eğitim için X girişlerinin beklenen çıktıları.

    # MLPClassifier -> multi-layer perceptron (MLP)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(2), random_state=1)

    # Eğitim verileri ile eğitim:
    clf = clf.fit(X, y)

    return clf

def tahmin(clf, X):
    # clf => Daha önceden eğitilmiş sınflandırıcı.
    # X => Sinir ağlarındaki giriş değerleri.
    return clf.predict(X)
