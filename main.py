from model.kmodes import kmodes
import numpy as np

def main():
    myModes = kmodes()

    points = np.random.rand(100, 5)

    centers = myModes.optimize(points)

    print(centers)


if __name__ == '__main__':
    main()