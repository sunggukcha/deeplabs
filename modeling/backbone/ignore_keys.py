

def ignore(pretrained, word):
	res = pretrained.copy()
	for k, v in pretrained.items():
		if word in k:
			del res[k]
	return res
