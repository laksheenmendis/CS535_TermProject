train_entries = open('../dataset/train_val_list.txt')
test_entries = open('../dataset/test_list.txt')

train_list = []
line = train_entries.readline()

while line != "":
    train_list.append(line.strip())
    line = train_entries.readline()

test_list = []
line = test_entries.readline()
while line != "":
    test_list.append(line.strip())
    line = test_entries.readline()



print(len(train_list))
print(len(test_list))

all_images_path = 'all_images'
train_dir_path = 'train'
test_dir_path = 'test'

script = '#!/bin/bash\n'
for image in train_list:
    script += 'cp ' + all_images_path + '/' + image + ' ' + train_dir_path + ';\n'

script += '\n'

for image in test_list:
    script += 'cp ' + all_images_path + '/' + image + ' ' + test_dir_path + ';\n'


outputFile = open('organize-train-test-images.sh', 'w')

outputFile.write(script)
outputFile.flush()
outputFile.close()

