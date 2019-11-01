
fs=open('new.txt','w+')
f1=open('file1.txt','r')
f2=open('file2.txt','r')
word1=f1.readline()
word2=f2.readline()
while word1:
    word1=word1[:-1]
    word2 = word2[:-1]
    fs.write('train_images/'+word1+'	train_labels/'+word2+'\n')
    word1=f1.readline()
    word2=f2.readline()

