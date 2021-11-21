from math import radians, sin, cos, asin, sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import pandas as pd


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Compute distance between two pairs of coordinates (lon1, lat1, lon2, lat2)
    See - (https://en.wikipedia.org/wiki/Haversine_formula)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2 * 6371 * asin(sqrt(a))


def return_significative_coef(model):
    """
    Returns p_value, lower and upper bound coefficients
    from a statsmodels object.
    """
    # Extract p_values
    p_values = model.pvalues.reset_index()
    p_values.columns = ['variable', 'p_value']

    # Extract coef_int
    coef = model.params.reset_index()
    coef.columns = ['variable', 'coef']
    return p_values.merge(coef,
                          on='variable')\
                   .query("p_value<0.05").sort_values(by='coef',
                                                      ascending=False)


def plot_kde_plot(df, variable, dimension):
    """
    Plot a side by side kdeplot for `variable`, split
    by `dimension`.
    """
    g = sns.FacetGrid(df, hue=dimension, col=dimension)
    g.map(sns.kdeplot, variable)


def standardize(df, features):
    df_standardized = df.copy()
    for f in features:
        mu = df[f].mean()
        sigma = df[f].std()
        df_standardized[f] = df[f].map(lambda x: (x - mu) / sigma)
    return df_standardized


def vif_table(df, features, with_standardize=False):
    if with_standardize:
        df = standardize(df, features)
    vif_table = pd.DataFrame()
    vif_table['index'] = [key for index, key in enumerate(df.keys())]
    vif_table['vif'] = [
        vif(df.values, index) for index, key in enumerate(df.keys())
    ]
    return vif_table


def super_subplot(df, target, figsize, graph_type):
    plt.figure(figsize=figsize)
    df_numerical = df[[
        key for key in df.keys()
        if df[key].dtypes == "float64" or df[key].dtypes == "int64"
    ]]

    if len(target) == 1 and df.shape[1] == 1:
        if graph_type == 'scatter':
            sns.scatterplot(data=df[target[0]])
        elif graph_type == 'hist':
            sns.histplot(data=df[target[0]], kde=True)
        elif graph_type == 'box':
            sns.boxplot(data=df[target[0]])
        elif graph_type == 'count':
            sns.countplot(data=df[target[0]])

    elif type(target) == list and len(target) == df.shape[1]:
        if graph_type == 'scatter':
            for i, v in enumerate(target):
                plt.subplot(len(target) // 3 + len(target) % 3, 3, i + 1)
                sns.scatterplot(data=df[[v]])

        elif graph_type == 'hist':
            for i, v in enumerate(target):
                plt.subplot(len(target) // 3 + len(target) % 3, 3, i + 1)
                sns.histplot(data=df[[v]], kde=True)

        elif graph_type == 'box':
            for i, v in enumerate(target):
                plt.subplot(len(target) // 3 + len(target) % 3, 3, i + 1)
                sns.boxplot(data=df[[v]])

        elif graph_type == 'count':
            for i, v in enumerate(target):
                plt.subplot(len(target) // 3 + len(target) % 3, 3, i + 1)
                sns.countplot(data=df[[v]])

    else:
        if graph_type == 'scatter':
            for i, v in enumerate(
                [key for key in df_numerical.keys() if key not in [target]]):
                plt.subplot(
                    len(df_numerical.keys()) // 3 +
                    len(df_numerical.keys()) % 3, 3, i + 1)
                sns.scatterplot(data=df, x=target, y=v)

        elif graph_type == 'hist':
            for i, v in enumerate(
                [key for key in df_numerical.keys() if key not in [target]]):
                plt.subplot(
                    len(df_numerical.keys()) // 3 +
                    len(df_numerical.keys()) % 3, 3, i + 1)
                sns.histplot(data=df, x=target, kde=True)

        elif graph_type == 'box':
            for i, v in enumerate(
                [key for key in df_numerical.keys() if key not in [target]]):
                plt.subplot(
                    len(df_numerical.keys()) // 3 +
                    len(df_numerical.keys()) % 3, 3, i + 1)
                sns.boxplot(data=df, x=target)

        elif graph_type == 'count':
            for i, v in enumerate(
                [key for key in df_numerical.keys() if key not in [target]]):
                plt.subplot(
                    len(df_numerical.keys()) // 3 +
                    len(df_numerical.keys()) % 3, 3, i + 1)
                sns.countplot(data=df, x=target)


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


def plot_loss_accuracy(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()
