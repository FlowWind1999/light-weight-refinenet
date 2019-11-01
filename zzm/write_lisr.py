fs=open('train.txt','w+')
fv=open('val.txt','w+')
for i in range(2807):
    if i<10:
        fs.write('train_images/00000{a}.png	train_labels/00000{a}.png\n'.format(a=i))
        fv.write('val_images/00000{a}.png	val_labels/00000{a}.png\n'.format(a=i))
    elif i<100:
        fs.write('train_images/0000{a}.png	train_labels/0000{a}.png\n'.format(a=i))
        fv.write('val_images/0000{a}.png	val_labels/0000{a}.png\n'.format(a=i))
    elif i<1000:
        fs.write('train_images/000{a}.png	train_labels/000{a}.png\n'.format(a=i))
    else:
        fs.write('train_images/00{a}.png	train_labels/00{a}.png\n'.format(a=i))





