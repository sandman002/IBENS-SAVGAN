import torch
import matplotlib.pyplot as plt
import math
import numpy as np

def plot_attn(at, at2, save_loc, step,suffix):
	at = at.squeeze()
	at = at.data.cpu().numpy()


	at2 = at2.squeeze()
	at2 = at2.data.cpu().numpy()
	# norm_= (np.linalg.norm(at - at2))
	print(sum(at[1,:]), sum(at2[1,:]))


	f, axarr = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), dpi=80, sharex=False, sharey=False)
	im1 = axarr[0].imshow(at, cmap='hot', interpolation='nearest')
	im2 = axarr[1].imshow(at2, cmap='hot', interpolation='nearest')

	axarr[0].title.set_text('Full Attn')
	axarr[1].title.set_text('NysApprox Attn')

	plt.colorbar(im1, ax=axarr[0])
	plt.colorbar(im2, ax=axarr[1])
	plt.savefig(save_loc+'/'+'step_'+str(step)+suffix+'.png')
	plt.clf()
	plt.close('all')