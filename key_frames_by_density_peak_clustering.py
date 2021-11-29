import numpy as np
import cv2
import time
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from collections import deque
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.neighbors import DistanceMetric
from kneed import KneeLocator

def get_video_fps(video):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)

    return round(fps) 

def get_video_resolution(video):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver)  < 3 :
        pass
    else :
        resolution = [
            video.get(cv2.CAP_PROP_FRAME_WIDTH),
            video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ]

    return resolution

# Not used
# def get_scaling_factor(resolution):
#     return round(resolution[0] / 360)

h_bins = [[np.min(x), np.max(x)] for x in np.array_split(range(0, 180), 12)]
s_bins = [[np.min(x), np.max(x)] for x in np.array_split(range(0, 255), 5)]
v_bins = s_bins

def quantize_frame(frame):
    start_time = time.time()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = np.zeros(76)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            h, s, v = hsv[i][j]
            index = 0
            
            # Для каждого пикселя вычиляем значение по формуле:
            # 5 * H_i + 3 * S_i + 3 * V_i , где "*_i" - индекс отрезка в котором лежит 
            # значение. Результат - 0...75. 
            for i1, bin in enumerate(h_bins):
                if (h >= bin[0] and h <= bin[1]):
                    index += 5*i1
                    break

            for i2, bin in enumerate(s_bins):
                if (s >= bin[0] and s <= bin[1]):
                    index += 3*i2
                    break

            for i3, bin in enumerate(v_bins):
                if (v >= bin[0] and v <= bin[1]):
                    index += 2*i3
                    break
                    
            # собираем результирующий вектор длины 76, где для каждого значения
            # из диапазона 0...75 - число пикселей с таким значением
            hist[index] += 1

    finish = time.time
    # print("{} seconds".format(time.time() - start_time)) # DEBUG


    return hist

def read_video(path):    
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print('Unable to open file')
        exit(0)

    frame_number = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # factor = 1.0 / get_scaling_factor(get_video_resolution(video))

    while True:
        curr_frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
        # print("{}/{}".format(frame_number, curr_frame_number)) # DEBUG

        ret, frame = video.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.resize(frame, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
        quantize_frame(frame)
        
        
        if curr_frame_number >= 300:
            break
        
    video.release()

# Попытка повысить время обработки видео за счет асинхронной обработки кадров видео. Не используется
# т.к. при привело в снижению времени обработки
def async_test(path):
    video = cv2.VideoCapture(path)

    if not video.isOpened():
        print('Unable to open file')
        exit(0)

    threadn = cv2.getNumberOfCPUs()
    print(threadn)
    pool = ThreadPool(processes = threadn)
    pending = deque()

    while True:
        while len(pending) > 0 and pending[0].ready():
            res = pending.popleft().get()
        if len(pending) < threadn:
            _ret, frame = video.read()
            # curr_frame_number = video.get(cv2.CAP_PROP_POS_FRAMES) # DEBUG

            frame = cv2.resize(frame, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
            task = pool.apply_async(quantize_frame, args=(frame.copy(),))
            pending.append(task)

    print('Done')

# Три функции ниже относятся к параллельной обработке видео. 
# Скорость отработки увеличилась - приближительно в 2 раза, но не кратно количеству ядер


def process_frame(group_number):
    video = cv2.VideoCapture(path)

    video.set(cv2.CAP_PROP_POS_FRAMES, group_number * group_size)

    frame_number = 0

    res_list = []
    while frame_number < group_size:
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
        # if frame_number == 0: # DEBUG
        #     cv2.imwrite("./frame.jpeg", frame)
        res_list.append(quantize_frame(frame))
        frame_number += 1

    f = open(str(group_number) + ".pickle", "wb")
    f.write(pickle.dumps(res_list))
    f.close()
    video.release()

def combine():
    """ Объединение результатов параллельной обработки """"
    pickles = []
    for file in os.listdir("./"):  
        if "pickle" in file:
            pickles.append(file)
        
    pickles = sorted(pickles, key=lambda x: int(x.split('.')[0]))
    # print(pickles) # DEBUG

    res_list = []
    for p in pickles:
        res = pickle.loads(open(p, "rb").read())
        res_list.extend(res)
    # print(len(res_list)) # DEBUG

    for p in pickles:
        os.remove(p)

    return res_list

def process_video(dump_file_name):
    """ Формарование вектора для дальнейшей работы. 
    Т.к. это самая длительная часть процесса, его результат сохраняется на диск с именем "dump_file_name.npy"
     """
    pool = mp.Pool(thread_number)
    pool.map(process_frame, range(thread_number))
    res = combine()
    print(len(res[0]))
    np.save(dump_file_name, res)

# Модифицированный DBSCAN для кластеризации кадров
class DBSCAN():
    def __init__(self, eps=1, min_samples=30, dist_function="euclidean"):
        self.eps = eps
        print("self.eps", self.eps, "self.min_samples", min_samples)
        self.min_samples = min_samples
        self.dist = DistanceMetric.get_metric(dist_function)

    def get_neighbors(self, sample_i):
        """ Считаем плотность каждого кадра. В качестве окрестности каждого кадра выбираем секунду в каждую сторону """
        if sample_i - fps < 0:
            left_samples = self.X[:sample_i]
            left_indexes = np.arange(0, sample_i)
            # print("left_samples = {}:{}".format(0, sample_i)) # DEBUG
        else:
            left_samples = self.X[sample_i - fps:sample_i]
            left_indexes = np.arange(sample_i - fps, sample_i)
            # print("left_samples = {}:{}".format(sample_i - fps, sample_i)) # DEBUG

        if sample_i + fps + 1 >= len(self.X):
            right_samples = self.X[sample_i + 1:]
            right_indexes = np.arange(sample_i + 1, len(self.X))
            # print("right_samples = {}:{}".format(sample_i + 1, len(self.X) - 1)) # DEBUG
        else:
            right_samples = self.X[sample_i + 1: sample_i + fps + 1]
            right_indexes = np.arange(sample_i + 1, sample_i + fps + 1)
            # print("right_samples = {}:{}".format(sample_i + 1, sample_i + fps + 1)) # DEBUG

        indexes = np.concatenate([left_indexes, right_indexes])
        X = np.concatenate([left_samples, right_samples])
        distances = self.dist.pairwise(X, [self.X[sample_i]])       
        neighbors = [indexes[i] for i in range(len(distances)) if distances[i] <= self.eps]
       
        return np.array(neighbors)

    def expand_cluster(self, sample_i, neighbors):
        """" Рекурсирное расширение кластера """
        cluster = [sample_i]
        # Iterate through neighbors
        for neighbor_i in neighbors:
            if not neighbor_i in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                self.neighbors[neighbor_i] = self.get_neighbors(neighbor_i)
                if len(self.neighbors[neighbor_i]) >= self.min_samples:
                    expanded_cluster = self.expand_cluster(
                        neighbor_i, self.neighbors[neighbor_i])
                    cluster = cluster + expanded_cluster
                else:
                    cluster.append(neighbor_i)
        return cluster

    def label_clusters(self):
        """ Разметка индексов кадров по кластерам. Помечаем -1 выбросы """
        self.labels = np.full(shape=self.X.shape[0], fill_value=-1)
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                self.labels[sample_i] = cluster_i
        return self.labels

    def get_cluster_point_density(self, sample_i, cluster):
        """ Считаем плотность точки уже внутри кластера """
        neighbors = [self.X[i] for i in cluster if i != sample_i]
        indexes = [i for i in cluster if i != sample_i]
        distances = self.dist.pairwise(neighbors, [self.X[sample_i]]) 
        res_list = [indexes[i] for i in range(len(distances)) if distances[i] <= self.eps]
        return len(res_list)

    def get_cluster_centers(self):  
        """ Находим в кластере точку масимальной плотности """
        self.cluster_centers = []      
        for cluster in self.clusters:
            max_density = 0
            max_density_i = -1
            for i in cluster:
                density = self.get_cluster_point_density(i, cluster)
                if density > max_density:
                    max_density = density
                    max_density_i = i
                
            self.cluster_centers.append(i)
    
    def get_n_frames_in_cluster(self, n):
        """ Возвращает n соседей центра кластера"""
        n_frames = []
        for i, cluster in enumerate(self.clusters):
            cluster.sort()
            center_i = cluster.index(self.cluster_centers[i])

            if center_i - n < 0:
                left_index = 0
            else:
                left_index = center_i - n

            if center_i + n + 1 >= len(cluster):
                right_index = len(cluster) - 1
            else:
                right_index = center_i + n + 1

            n_frames.append(cluster[left_index:right_index])

        return n_frames

    def fit(self, X):
        """ Кластеризация """
        self.X = X
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}
        n_samples = np.shape(self.X)[0]
        # Iterate through samples and expand clusters from them
        # if they have more neighbors than self.min_samples
        for sample_i in range(n_samples):
            if sample_i in self.visited_samples:
                continue
            self.neighbors[sample_i] = self.get_neighbors(sample_i)
            # print(sample_i, len(self.neighbors[sample_i]))
            if len(self.neighbors[sample_i]) >= self.min_samples:
                # print(sample_i)
                # If core point => mark as visited
                self.visited_samples.append(sample_i)
                # Sample has more neighbors than self.min_samples => expand
                # cluster from sample
                new_cluster = self.expand_cluster(
                    sample_i, self.neighbors[sample_i])
                # Add cluster to list of clusters
                self.clusters.append(new_cluster)

        # Get the resulting cluster labels
        clusters = self.label_clusters()
        return clusters

    # def test(self, i, j): # DEBUG
    #     video = cv2.VideoCapture(path)
    #     video.set(cv2.CAP_PROP_POS_FRAMES, i)
    #     ret, frame = video.read()
    #     frame = cv2.resize(frame, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
    #     cv2.imwrite('frame{:d}.jpg'.format(i), frame)
    #     video.set(cv2.CAP_PROP_POS_FRAMES, j)
    #     ret, frame = video.read()
    #     frame = cv2.resize(frame, None, fx=0.05, fy=0.05, interpolation=cv2.INTER_AREA)
    #     cv2.imwrite('frame{:d}.jpg'.format(j), frame)
    #     video.release()
    #     print(self.dist.pairwise([self.X[i]], [self.X[j]]))

    def save_n_key_frames(self, n=10):
        """ Сохраняем центр каждого кластера и n соседей на диск.
        
        Создает в текущей директории папку ./frames и вней папку c номером каждого кластера
        """
        video = cv2.VideoCapture(path)

        n_frames = self.get_n_frames_in_cluster(n)
        frames_indexes = [frame for i, cluster in enumerate(n_frames) for frame in cluster]
        frames_clusters = [i for i, cluster in enumerate(n_frames) for frame in cluster]
        # print(frames_indexes) # DEBUG
        # print(frames_clusters) # DEBUG
        
        frames_path = "./frames"
        if not os.path.exists(frames_path):
            os.mkdir(frames_path)

        while True:
            ret, frame = video.read()

            if not ret:
                break

            frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
            # print(str(frame_number) + '\r) # DEBUG

            if frame_number in frames_indexes:
                cluster_path = frames_path + "/" + str(frames_clusters[frames_indexes.index(frame_number)])
                
                if not os.path.exists(cluster_path):
                    os.mkdir(cluster_path)

                # print(cluster_path + "/" + str(frame_index) + ".jpeg") # DEBUG
                cv2.imwrite(cluster_path + "/" + str(frame_number) + ".jpeg", frame)

        video.release()

# DEBUG
# def save_all_frames():
#     video = cv2.VideoCapture(path)

#     frame_index = 0
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break

#         cluster_path = "./all"
            
        
#         if not os.path.exists(cluster_path):
#             os.mkdir(cluster_path)

#         cv2.imwrite(cluster_path + "/" + str(frame_index) + ".jpeg", frame)

#         frame_index += 1

path = "/home/roman/Drive/ITMO/Palantir/video.mp4"
video = cv2.VideoCapture(path)
thread_number = cv2.getNumberOfCPUs()
frame_number = video.get(cv2.CAP_PROP_FRAME_COUNT)
group_size = frame_number // thread_number
fps = get_video_fps(video)
video.release()

def main():
    start = time.time()
    
    dump_file_name = "dump.npy"
    # Считаем вектро по видео. Вызов фукции можно закомментировать при повторных запусках 
    process_video(dump_file_name)
    # Читаем посчитанный вектор
    data = np.load(dump_file_name)

    # Находим оптимальное значение epsilon для DBSCAN

    # Cчитаем расстояние между соседними точками
    neigh = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)      

    # Сортируем массив
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    # Находим точку изгиба
    kl = KneeLocator(np.arange(len(distances)), distances, curve="convex")
    eps = distances[kl.knee]//2 # Магическое число! По итогам обработки трех разных видео заметил, что kl.knee задирается вправо

    method = DBSCAN(eps=eps, min_samples=20)

    # Выполняем кластеризацию
    method.fit(data) 

    # Находим центры кластеров
    method.get_cluster_centers()

    # Cохраняем на диск кадры центров и n соседей
    method.save_n_key_frames()

    print("Total execution time: {}".format(time.time() - start))

if __name__ == "__main__":
    main()