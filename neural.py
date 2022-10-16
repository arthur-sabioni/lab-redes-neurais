import pandas as pd
import numpy as np
import math

def degrau(x):
    x = x[0]
    if x >=0:
        return 1
    else:
        return 0

def sigmoidal(x):
    x = x[0]
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(-x))

def perceptron(max_it, alpha, df, function):
    W = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    b = np.array([[0], [0], [0]])
    Ep = []
    t=1
    E=1
    while(t < max_it and E > 0):
        E = 0
        for i in range(0, len(df)):
            x = np.transpose([df.values[i][0:6]])
            y = W.dot(x) + b
            y = np.array([list(map(function, y))]).T
            e = df.values[i][6] - y
            W = W + alpha*e.dot(x.T)
            b = b + alpha*E
            E = E + e.T.dot(e)[0][0]
        Ep.append(E)
        t += 1
    return W, b

def perceptron_test(df, W, b, function):
    hits = 0
    for i in range(0, len(df)):
        x = np.transpose([df.values[i][0:6]])
        y = W.dot(x) + b
        y = np.array([list(map(function, y))]).T
        e = df.values[i][6] - y
        if(np.all(e == 0)):
            hits += 1
    return hits*100/len(df)

if __name__ == "__main__":

    # Seed to randomize dfs
    seed = 3

    df = pd.read_csv("./dataframe.csv")

    # Separate per class
    df_1 = df[df['class'] == 'DH']
    df_2 = df[df['class'] == 'SL']
    df_3 = df[df['class'] == 'NO']

    # Slice random 66% samples
    df_1_samples = df_1.sample(n=int(len(df_1)*2/3), random_state=seed)
    df_2_samples = df_2.sample(n=int(len(df_2)*2/3), random_state=seed)
    df_3_samples = df_3.sample(n=int(len(df_3)*2/3), random_state=seed)

    # Get remaining 33% samples
    df_1 = pd.concat([df_1, df_1_samples]).drop_duplicates(keep=False)
    df_2 = pd.concat([df_2, df_2_samples]).drop_duplicates(keep=False)
    df_3 = pd.concat([df_3, df_3_samples]).drop_duplicates(keep=False)

    # Concatenate traning and tests samples
    df_training = pd.concat([df_1_samples, df_2_samples, df_3_samples])
    df_tests = pd.concat([df_1, df_2, df_3])

    # Filtering data
    translate_dict = {
        'DH': [[0], [0], [1]],
        'SL': [[0], [1], [0]],
        'NO': [[1], [0], [0]],
    }
    df_tests['class'] = df_tests['class'].apply(lambda x: translate_dict[x])
    df_training['class'] = df_training['class'].apply(lambda x: translate_dict[x])

    # Shuffle data
    df_tests = df_tests.sample(frac=1, random_state=seed)
    df_training = df_training.sample(frac=1, random_state=seed)

    weight, bias = perceptron(5, 0.1, df_training, degrau)

    print("Treinamento Concluído")
    print("Peso: \n", weight)
    print("Bias: \n", bias)

    accuracy = perceptron_test(df_tests, weight, bias, degrau)

    print("Teste Concluído")
    print("Acurracy: %.2f" % accuracy)