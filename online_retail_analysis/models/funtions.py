from sklearn.cluster import MiniBatchKMeans as KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def train_elbow_cluster(train,start=1,finish=10,each=1):
    wcss = {} #dicionario vacio para guardar los errores
    for i in range(start,finish,each):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(train)
        wcss[i] = kmeans.inertia_ #suma de distancias cuadradas a los centros del grupos más cercanos
    plt.title('Metodo de Elbow ')
    plt.xlabel('k= n grupos')
    plt.ylabel('WCSS')
    sns.pointplot(x=list(wcss.keys()), y=list(wcss.values()))
    plt.show()

def Kmeans(train, clusters_number, original_df_rfm):
    kmeans = KMeans(n_clusters = clusters_number, random_state = 42)
    # Predict the cluster
    y_kmeans = kmeans.fit_predict(train)
    # Create a cluster label column in original dataset
    df_cluster = original_df_rfm.assign(Cluster = y_kmeans)
    return df_cluster