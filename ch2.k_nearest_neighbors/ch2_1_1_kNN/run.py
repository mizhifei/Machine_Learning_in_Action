import kNN

group, labels = kNN.create_data_set()

print(group)

print(labels)


test = kNN.classify0([0,0], group, labels, 3)

print(test)