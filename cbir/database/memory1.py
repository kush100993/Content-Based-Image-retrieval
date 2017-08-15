import sys
sys.path.append("..")
from cbir.utils import load, dump, delete
from scipy.cluster.vq import vq,whiten
import numpy as np

# to compute square distance between query descriptor and the codebook
def compute_sq_dist(codebook,query_disc):
	sq_dist=[]
	for i in range(len(query_disc)):
		v = codebook[i]-query_disc[i]
		v = np.sum(v*v,axis=1)
		sq_dist.append(v)
	return sq_dist


class vlad_database():
	def __init__(self,database_path = None):
		self.image_paths = []
		self.codes = []
		self.code_book = []
		self.database_path = database_path
		self.load_from_disk()


	def load_from_disk(self):
		data = load(self.database_path)
		if (data):
			self.image_paths = data['images']
			self.codes = data['code']
			self.code_book = data['codebook']
		else:
			print('No Database created, please create a new one')

	def reset(self):
		self.image_paths=[]
		self.codes=[]
		self.code_book=[]

	def save_to_disk(self):
		data = {'codebook':self.code_book, 'code': self.codes, 'images': self.image_paths}
		dump(data, self.database_path)

	def __assert_codebook(self):
		if not self.code_book:
			raise Exception('Codebook is empty,please generate one.')


	def delete_vlad_database(self):
		delete(self.database_path)


	def insert(self, image, single_des = None):
		self.image_paths.append(image)
		if single_des:
			self.__assert_codebook()
			co = list(zip(*map(vq,single_des,self.code_book)))
			code = list(co[0])
			self.codes = list(map(np.append,self.codes,code))
		self.save_to_disk()


	# serach for the k closest matches of the given query image 
	def query(self, query_disc, k = 10):
		self.__assert_codebook()
		sq_dist = compute_sq_dist(self.code_book,query_disc)
		val=[]
		for i in range(self.codes[0].shape[0]):
			dist = 0
			for j in range(len(sq_dist)):
				dist = dist + sq_dist[j][self.codes[j][i]]
			val.append(dist)
		val = np.asarray(val)
		val = np.sqrt(val)
		n_images = list(zip(val,self.image_paths))
		n_images = sorted(n_images)
		#print(n_images)
		return n_images[:k]	


