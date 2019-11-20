import torch
import math
import numpy
import scipy.stats
import statsmodels.stats.proportion
def random_mask_batch_one_sample(batch, keep_per_image, reuse_noise = False):
	batch = batch.permute(0,2,3,1) #color channel last
	flat = batch.reshape(batch.shape[0], batch.shape[1]* batch.shape[2],  batch.shape[3]) #batch*pixels*channels
	out_c1 = torch.zeros(flat.shape).cuda()
	out_c2 = torch.zeros(flat.shape).cuda()
	if (reuse_noise):
		#ones = torch.ones(flat.shape[1]).cuda()
		#idx = torch.multinomial(ones,keep_per_image)
		idx = torch.tensor(numpy.random.choice(flat.shape[1],keep_per_image,replace=False)).cuda()
		out_c1[:,idx] = flat[:,idx]
		out_c2[:,idx] = 1.-flat[:,idx]
	else:
		idx = batch_choose(flat.shape[1],keep_per_image, flat.shape[0])
		idx_range = torch.arange(idx.shape[0]).cuda().unsqueeze(0).t()
		out_c1[(idx_range,idx)] = flat[(idx_range,idx)]
		out_c2[(idx_range,idx)] = 1.-flat[(idx_range,idx)]
		#print(flat[0])
		#print(idx[0])
		#print(out_c1[0])
		#print(out_c2[0])
	out_c1 = out_c1.reshape(batch.shape).permute(0,3,1,2)
	out_c2 = out_c2.reshape(batch.shape).permute(0,3,1,2)
	out = torch.cat((out_c1,out_c2), 1)
	#print(out[14,:,5:10,5:10])
	return out

#binom test(nA, nA + nB, p)
def avg_hard_forward(batch, net, num_samples, keep, sub_batch = 0):
	if (sub_batch == 0):
		expanded = batch.repeat_interleave(num_samples,0) # shape: batch*num_samples, etc
		masked = random_mask_batch_one_sample(expanded, keep)
		soft = net(masked)
		votes = soft.max(1)[1]
		hard = torch.zeros(soft.shape).cuda()
		hard.scatter_(1,votes.unsqueeze(1),1)
		return hard.reshape((batch.shape[0],num_samples,) + hard.shape[1:]).mean(dim=1)
	else:
		aresums = False
		count = num_samples
		while (count > 0):
			curr_sub_batch = min(count, sub_batch)
			count -= sub_batch
			expanded = batch.repeat_interleave(curr_sub_batch,0) # shape: batch*num_samples, etc
			masked = random_mask_batch_one_sample(expanded, keep)
			soft = net(masked)
			votes = soft.max(1)[1]
			hard = torch.zeros(soft.shape).cuda()
			hard.scatter_(1,votes.unsqueeze(1),1)
			if (aresums == False):
				aresums = True
				sums = torch.zeros((batch.shape[0],) +  hard.shape[1:]).cuda()
			sums += hard.reshape((batch.shape[0],curr_sub_batch,) + hard.shape[1:]).sum(dim=1)
		return sums/num_samples
def lc_bound(k, n ,alpha):
	return statsmodels.stats.proportion.proportion_confint(k, n, alpha=2*alpha, method="beta")[0]
# returns -1 for incorrect, 0 for correct but no certificate, positive otherwise
def certify(batch, labels, net, alpha, keep, num_samples_select, num_samples_bound, sub_batch = 0 ):
	guesses = avg_hard_forward(batch, net, num_samples_select, keep,sub_batch=sub_batch).max(1)[1]
	bound_scores = avg_hard_forward(batch, net, num_samples_bound, keep,sub_batch=sub_batch)		
	bound_selected_scores = torch.gather(bound_scores,1,guesses.unsqueeze(1)).squeeze(0)
	if(len(bound_selected_scores.shape) == 1):
		bound_selected_scores = bound_selected_scores.unsqueeze(0)
	bound_selected_scores = lc_bound((bound_selected_scores*num_samples_bound).cpu().numpy(),num_samples_bound,alpha)
	radii = population_radius_for_majority(bound_selected_scores, batch[0,0].nelement(), keep)
	radii[guesses != labels] = -1
	return radii
def predict(batch, net ,keep, num_samples, alpha, sub_batch = 0 ):
	scores = avg_hard_forward(batch, net, num_samples, keep,sub_batch=sub_batch)
	toptwo = torch.topk(scores.cpu(),2,sorted=True)
	toptwoidx = toptwo[1]
	toptwocounts = toptwo[0]*num_samples
	out = -1* torch.ones(batch.shape[0], dtype = torch.long)
	tests = numpy.array([scipy.stats.binom_test(toptwocounts[idx,0],toptwocounts[idx,0]+toptwocounts[idx,1], .5) for idx in range(batch.shape[0])])
	out[tests <= alpha] = toptwoidx[tests <= alpha][:,0]
	return out
#((size-r)! / (keep! * (size-r-keep!))) / (size!/ (keep! * (size-keep)!))

#((size-r)! (size-keep)!/ ( size! * size-r-keep!)) 

#1.5 - ((size- r) choose keep)/ (size choose keep) < score
# 1.5-score  < ((size- r) choose keep) /(size choose keep)
def population_radius_for_majority(scores_of_true, size, keep):
	count = scores_of_true.shape[0]
	done = torch.zeros(count, dtype=torch.uint8)
	radii = torch.zeros(count, dtype=torch.long)
	radius = 0
	lhs = (1.5 - scores_of_true ).squeeze(1)
	#print(lhs)
	while (done.sum() < count):

		rhs = math.factorial(size-radius) * math.factorial(size-keep)/ (math.factorial(size) * math.factorial(size-keep-radius))
		done[torch.tensor(lhs >= rhs)] = 1
		radii[torch.tensor(lhs < rhs)] = radius
		radius += 1
	return radii


def batch_choose_fake(n,k,batches):
	start = torch.cuda.Event(enable_timing=True)
	end = torch.cuda.Event(enable_timing=True)
	start.record()
	ones = torch.ones(n).cuda()
	out = torch.stack([torch.multinomial(ones,k) for x in range(batches)])
	end.record()
	torch.cuda.synchronize()
	print(start.elapsed_time(end))
	return out

def batch_choose(n,k,batches):
	#start = torch.cuda.Event(enable_timing=True)
	#end = torch.cuda.Event(enable_timing=True)
	#start.record()
	out = torch.zeros((batches,k), dtype=torch.long).cuda()
	for i in range(k):
		out[:,i] = torch.randint(0,n-i, (batches,))
		if (i != 0):
			last_boost = torch.zeros(batches, dtype=torch.long).cuda()
			boost = (out[:,:i] <=(out[:,i]+last_boost).unsqueeze(0).t()).sum(dim=1)
			while (boost.eq(last_boost).sum() != batches):
				last_boost = boost
				boost = (out[:,:i] <=(out[:,i]+last_boost).unsqueeze(0).t()).sum(dim=1)
			out[:,i]  += boost
	#end.record()
	#torch.cuda.synchronize()
	#print(start.elapsed_time(end))
	return out
