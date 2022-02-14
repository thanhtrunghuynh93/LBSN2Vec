from scipy.io import loadmat

mat = loadmat('../dataset/cleaned_hongzhi.mat')
print(mat['selected_checkins'])