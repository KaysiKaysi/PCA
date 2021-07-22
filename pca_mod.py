import numpy as np

class PCA:

	def __init__(self, n_components):
		self.n_components = n_components
		self.components_ = None
		self.explained_variance_ = None
		self.explained_variance_ratio_ = None
		self.mean_ = None
		self.covariance_ = None

	def fit(self, input_matrix):
		# centering with mean
		# for simple use can also apply np.mean(input_matrix, axis = 0)
		sum = 0
		mean_lst = []
		for i in range(input_matrix.shape[1]):
			sum = 0
			for j in range(input_matrix.shape[0]):
				sum = sum + input_matrix[j][i]
			mean = sum/input_matrix.shape[0]
			mean_lst.append(mean)
		self.mean_ = np.asarray(mean_lst)
		input_matrix_pr = input_matrix - self.mean_
		# covariance
		# for simple use can apply np.cov(input_matrix.T)
		self.covariance_ = np.dot(input_matrix_pr.T,input_matrix_pr)/(input_matrix_pr.shape[0] - 1)
		# eigenvalues, eigenvectors
		eig_val, eig_vec = np.linalg.eig(self.covariance_)
		# sort eigenvalues, eigenvectors
		eig_vec = eig_vec.T
		idx = np.argsort(eig_val)[::-1]
		eig_val_sorted = eig_val[idx]
		eig_vec_sorted = eig_vec[idx]
		# store first n eigenvalues, eigenvectors
		self.explained_variance_ = eig_val_sorted[0:self.n_components]
		self.components_ = eig_vec_sorted[0:self.n_components]
		# calculate eigenvalue ratio
		apply_ratio = lambda x : x/self.explained_variance_.sum()
		var_ratio = np.vectorize(apply_ratio)
		self.explained_variance_ratio_ = var_ratio(self.explained_variance_)

	def transform(self, input_matrix):
		# project matrix
		input_matrix_pr = input_matrix - self.mean_
		tr_matrix = np.dot(input_matrix_pr, self.components_.T)
		return tr_matrix

	def inverse_transform(self, tr_matrix):
		# revert from transformation to initial matrix
		input_matrix = np.dot(tr_matrix, self.components_) + self.mean_
		return input_matrix