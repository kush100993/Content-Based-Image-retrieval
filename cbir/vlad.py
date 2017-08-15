import cv2
import numpy as np
import os
import glob
from cbir import utils

from sklearn.decomposition import PCA
from cbir.database.memory1 import vlad_database
from cbir.kmeans.sklrn import KMeans
from scipy.cluster.vq import vq,kmeans2,whiten

sift = cv2.xfeatures2d.SIFT_create()


# Extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(image, None)
    if des is None:
        des = np.empty([0, 128], dtype='float32')
    return des


# [1, 1, 1, 2, 3, 3] => { 1: 3, 2: 1, 3: 2 }
def occurences(words):
    unique, counts = np.unique(words, return_counts=True)
    return dict(zip(unique, counts))


class VLADEngine():
    def __init__(self, data_directory='./data'):  #change
        self.data_directory = data_directory
        self.code_book=[]
        self.kmeans = KMeans(
            dictionary_path=self.datapath('dictionary.pickle')
        )

        self.vlad_database = vlad_database(database_path=self.datapath('database_vlad.pickle'))

    def datapath(self, filename):
        return os.path.join(self.data_directory, filename)

    def __assert_dictionary_exists(self):
        if not self.kmeans.dictionary_exists():
            raise Exception('Need to load or train dictionary')


    # Used to train for both BoW and VLAD
    def train(self, clusters=1000, limit=None, path='./images/flickr1k/'):  #change
        self.kmeans = KMeans(dictionary_size = clusters,dictionary_path=self.datapath('dictionary.pickle'))
        all_features = np.empty([0, 128], dtype='float32')
        i = 0
        for filepath in glob.iglob(os.path.join(path, '*.jpg')):
            i += 1
            if (i > limit):
                break
            features = extract_features(filepath)
            all_features = np.vstack((all_features, features))
            # print(all_features.shape)

        if len(all_features) == 0:
            raise Exception('No feature extracted from "%s"' % path)

        centroids = self.kmeans.fit(all_features)
        #self.centroids_vec = self.kmeans.centroid  #change
        self.kmeans.save(self.datapath('dictionary.pickle'))
        print("done")
        return centroids

    def _vlad(self,image):
        self.__assert_dictionary_exists()
        features = extract_features(image)
        words = np.asarray(self.kmeans.predict(features))    #lables of each individual feature of an image predicted by k-means clustring.
        centroid_vec = np.asarray(self.kmeans.centroid)
        v = []
        for i in range(centroid_vec.shape[0]):
            if(np.sum(words==i) > 0):
                v_i = np.sum(features[words==i,:]-centroid_vec[i],axis = 0)
                v.append(v_i)
            else:
                v.append(np.zeros([128],dtype = 'float32'))
        v = np.hstack(v)
        v = np.sign(v)*np.sqrt(np.abs(v))
        v = v/np.linalg.norm(v)                 #L2 normalization.
        return v        


    def apply_PCA(self,vlad_vector_data,components=128):
        vlad_d = vlad_vector_data
        pca = PCA(n_components = components)
        pca.fit(vlad_d)
        utils.dump(pca,'C:/Users/kush/Desktop/pca_transfrom.pkl')
        x1 = pca.transform(vlad_d)
        return x1


    def vlad_collection(self,no_of_images = 1000,path='./images/flickr1k/'):
        data=[]
        self.delete_vlad_database()
        self.vlad_database.reset()
        for i,image in enumerate(glob.glob(os.path.join(path, '*.jpg'))):
            if i<no_of_images:
                print(i)
                img_vec_vlad = self._vlad(image)
                data.append(img_vec_vlad)
                self.vlad_database.insert(image = image)
            else:
                break
        data = np.vstack(data)
        return data 


    def gen_codebook(self,Data):
        print('Creating new codebook')
        Data = np.split(Data,8,axis = 1)
        for sub_vec in Data:
            code_book,code = kmeans2(sub_vec,iter = 50,k = 256,minit = 'points')
            code,_ = vq(whiten(sub_vec), code_book)
            self.vlad_database.code_book.append(code_book)
            self.vlad_database.codes.append(code)
        self.vlad_database.save_to_disk()


    def _single_gen(self,path):
        single_descriptor = self._vlad(path)
        pca = utils.load('C:/Users/kush/Desktop/pca_transfrom.pkl')
        single_descriptor = pca.transform(single_descriptor.reshape(1,-1))
        s_d = np.split(single_descriptor,8,axis = 1)
        return s_d


    def insert(self, path):
        s_d = self._single_gen(path)
        self.vlad_database.insert(image = path,single_des = s_d)


    def Query(self,path,k):
        que = self._single_gen(path)
        matches = self.vlad_database.query(que,k)
        return matches


    def delete_vlad_database(self):
        return self.vlad_database.delete_vlad_database()
