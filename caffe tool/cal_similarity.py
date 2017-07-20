import numpy as np
import sys
import collections
from caffe.proto import caffe_pb2
from numpy import *
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import math
import random
import copy
import leveldb
import caffe
'''
feature_path
@name
@features

mean_file
@mean
@mean value
'''

INF_MAX = 9999999.999


def swap(x, y):
    return y, x


def hash_code_write(file_path, name, float_data):
    # write into leveldb file according to hash code
    feature_db = leveldb.LevelDB(file_path)
    feature_db.Put(str(name), str(float_data))


def hash_features_write(file_path, hash_set, hash_features):
    # write into leveldb file according to hash code    
    feature_db = leveldb.LevelDB(file_path)
    for key, value in hash_set.items():
        feature_db.Put(key, hash_features[value].SerializeToString())


def clear_folder(file_path):
    if (os.path.exists(file_path)):
        shutil.rmtree(file_path)
    os.mkdir(file_path)


# get hash code according mean features and src img features
def get_hash_code(mean, float_data, dimension):
    hash_code = ""
    if (len(mean) < dimension or len(float_data) < dimension):
        print('error: get hash code error')
        return hash_code

    for i in range(dimension):
        if (float_data[i] >= 0.5):
            hash_code += "1"
        else:
            hash_code += "0"
    return hash_code


# calculate distance between two hash codes
def cal_hash_code_distance(hash_code, lines, distance, top, dimension):
    num = len(lines)
    length = len(hash_code)
    index = 0
    similar_hash_codes = []
    hash_code_bit = []
    differences = []
    if (top < 1):
        return similar_hash_code

    for i in range(num):
        dif = 0
        if (dimension > len(lines[i]) or dimension > length):
            print('error: hash_code\'s length exceed lines[i]\'s length')
            break

        bits = ''
        for j in range(length):
            if (hash_code[j] != lines[i][j]):
                dif += 1
                bits += '1'
            else:
                bits += '0'

        if (index < top or dif <= distance):
            similar_hash_codes.append(lines[i])
            differences.append(dif)
            hash_code_bit.append(bits)
            index += 1
        elif differences[index - 1] <= dif:
            continue
        else:
            similar_hash_codes[index - 1] = lines[i]
            differences[index - 1] = dif
            hash_code_bit[index - 1] = bits

        if (index > 1):
            for j in range(len(differences) - 2, -1, -1):
                if (differences[j] > differences[j + 1]):
                    differences[j], differences[j + 1] = swap(differences[j], differences[j + 1])
                    similar_hash_codes[j], similar_hash_codes[j + 1] = swap(similar_hash_codes[j],
                                                                            similar_hash_codes[j + 1])
                    hash_code_bit[j], hash_code_bit[j + 1] = swap(hash_code_bit[j], hash_code_bit[j + 1])
                else:
                    break

    return similar_hash_codes, hash_code_bit


# convert string to float
def get_float_feature(string):
    value = []
    strs = string.split()
    for item in strs:
        lhs = 0
        rhs = len(item)
        if (rhs < 1):
            continue

        while (rhs > lhs and item[rhs - 1].isdigit() == False):
            rhs -= 1
        while (rhs > lhs and item[lhs].isdigit() == False):
            lhs += 1
        if (rhs > lhs):
            value.append(float(item[lhs:rhs - lhs]))
    return value


# calculate euler between two vector features
def calculate_euler(train_feature, test_feature, dimension):
    global INF_MAX
    diff = INF_MAX
    if (len(train_feature) < dimension or len(test_feature) < dimension):
        return diff

    diff = 0.0
    for i in range(dimension):
        val = train_feature[i] - test_feature[i]
        diff += val * val

    return math.sqrt(diff)


# calculate cos between two vector features
def calculate_cos(train_feature, test_feature, dimension):
    global INF_MAX
    diff = INF_MAX
    if (len(train_feature) < dimension or len(test_feature) < dimension):
        return diff

    XY = 0.0
    X = 0.0
    Y = 0.0
    for i in range(dimension):
        XY += train_feature[i] * test_feature[i]
        X += train_feature[i] * train_feature[i]
        Y += test_feature[i] * test_feature[i]

    return XY / (math.sqrt(X) * math.sqrt(Y) + 1e-5)


# calculate man between two vector features
def calculate_man(train_feature, test_feature, dimension):
    global INF_MAX
    diff = INF_MAX
    if (len(train_feature) < dimension or len(test_feature) < dimension):
        return diff

    diff = 0.0
    for i in range(dimension):
        diff += abs(train_feature[i] - test_feature[i])

    return diff


# calculate weight
def calculate_weight(feature, a, b, eps, dimension):
    weight = feature
    if (len(feature) < dimension):
        return weight

    # sum_s = 0.0
    # for i in range(dimension):
    #    sum_s += pow(feature[i], 1.0 / a)
    # sum_s = pow(sum_s, a)
    # for i in range(dimension):
    #    weight[i] *= pow(feature[i] / sum_s, 1.0 / b)

    # sum_s = eps
    # q = []
    # for i in range(dimension):
    #    val = 0.0
    #    if (feature[i] > 0.0):
    #       val = 1.0 / dimension
    #    q.append(val + eps)
    #    sum_s += val
    # for i in range(dimension):
    #    weight[i] *= math.log(sum_s / q[i])

    return weight


# get similar img's path
def get_img_name(similar_hash_codes, feature_path, test_feature, dimension, top, answer, kind):
    img_name = []
    similarity = []
    precision = 0
    index = 0
    if (top < 1):
        return img_name, similarity, precision

    # open features
    db = leveldb.LevelDB(feature_path)

    a = 0.5
    b = 1.0
    eps = 1e-5
    feature = copy.deepcopy(test_feature)
    test = calculate_weight(feature, a, b, eps, dimension)

    for item in similar_hash_codes:
        if (len(item) < dimension):
            print('item = ' + item + ', length < ' + str(dimension))
            continue

        value = db.Get(str(item[0:dimension]))
        datum = caffe_pb2.Datum.FromString(value)

        for img in datum.imgs:
            strs = img.name.split('/')
            if (len(strs) > 1 and strs[len(strs) - 2] == answer):
                precision = 1

            feature = copy.deepcopy(img.features)
            train = calculate_weight(feature, a, b, eps, dimension)
            diff = 0.0
            if (kind == 'euler'):
                # calculate Euler distance between two features
                diff = calculate_euler(train, test, dimension)
            elif (kind == 'cos'):
                # calculate cos between two features
                diff = calculate_cos(train, test, dimension)
            elif (kind == 'man'):
                # calculate cos between two features
                diff = calculate_man(train, test, dimension)
            else:
                print('kind error, euler, cos')
                return img_name, similarity, precision

            # select top N imgs which difference is lowest
            if (len(img_name) < top):
                img_name.append(img.name)
                similarity.append(diff)
            elif similarity[top - 1] > diff:
                img_name[top - 1] = img.name
                similarity[top - 1] = diff
            else:
                continue

            if (len(img_name) > 1):
                for i in range(len(img_name) - 2, -1, -1):
                    if (similarity[i] > similarity[i + 1]):
                        similarity[i], similarity[i + 1] = swap(similarity[i], similarity[i + 1])
                        img_name[i], img_name[i + 1] = swap(img_name[i], img_name[i + 1])
                    else:
                        break
        index += 1

    return img_name, similarity, precision


def getFeature(db_name, img_num, dimension, mean_file, hash_file, feature_path):
    # open leveldb files
    db = leveldb.LevelDB(db_name)
    it = db.RangeIter()
    features = empty([1, dimension])
    sum_features = []
    for i in range(dimension):
        sum_features.append(0.0)

    for key, value in it:
        datum = caffe_pb2.Datum.FromString(value)
        features[0] = datum.float_data
        sum_features += features[0]

    # write mean value into leveldb file and txt file
    clear_folder(mean_file)
    output = open(mean_file + '.txt', 'w')
    mean = sum_features / img_num
    hash_code_write(mean_file, 'mean_value', mean)
    output.write(str(mean) + "\n")
    output.close()
    # print(mean)

    # remove original features
    clear_folder(feature_path)

    hash_num = 0
    output = open(hash_file, 'w')
    it = db.RangeIter()
    hash_set = dict()
    hash_features = []
    # features_db = leveldb.LevelDB(feature_path)
    for key, value in it:
        datum = caffe_pb2.Datum.FromString(value)
        # get hash code
        hash_code = get_hash_code(mean, datum.float_data, dimension)
        # hash code dedup
        if (hash_code not in hash_set):
            hash_set[hash_code] = hash_num
            node = caffe_pb2.Datum()
            node.img_label = hash_code
            hash_features.append(node)
            output.write(hash_code + "\n")
            hash_num += 1

        # read feature into Img
        img = hash_features[hash_set[hash_code]].imgs.add()
        img.name = datum.name
        img.features.extend(datum.float_data)

    output.close()
    # write into leveldb file according to hash code
    hash_features_write(feature_path, hash_set, hash_features)
    print(hash_num)


def similarPicture(db_name, img_num, dimension, mean_file, hash_file, feature_path, distance, top, output_folder, count,
                   rate, write_into_file):
    # read mean value
    mean_value_db = leveldb.LevelDB(mean_file)
    mean = get_float_feature(mean_value_db.Get('mean_value'))
    # print(mean)

    clear_folder(output_folder)
    index = 0

    # read hash code
    fin = open(hash_file, 'r')
    lines = fin.readlines()
    fin.close()

    precision = 0
    precision_euler = 0
    precision_cos = 0
    precision_man = 0

    # read test dataset features
    db = leveldb.LevelDB(db_name)
    it = db.RangeIter()
    for key, value in it:
        # generate random digit
        rand = random.randint(1, 1000) / 1000.0
        if (rand > rate):
            continue
        # read src img features and name
        datum = caffe_pb2.Datum.FromString(value)
        strs = datum.name.split('/')
        answer = ''
        if len(strs) > 1:
            answer = strs[len(strs) - 2]
        # print('src img:')
        # print(datum.name)

        # write src img path into txt file
        if (write_into_file):
            output = open(output_folder + '/' + str(index) + '.txt', 'w')
            output.write(datum.name + '\n')

        # get hash code
        hash_code = get_hash_code(mean, datum.float_data, dimension)
        if (len(hash_code) != dimension):
            print('error: split hash code error')
            output.close()
            break

        # calculate two hash code's distance, select top best similar hash codes
        similar_hash_codes, hash_code_bit = cal_hash_code_distance(hash_code, lines, distance, top, dimension)

        # get image name
        imgs, similarity, geted = get_img_name(similar_hash_codes, feature_path, datum.float_data, dimension, top,
                                               answer, 'euler')
        # write similar img into file
        for img in imgs:
            strs = img.split('/')
            if (len(strs) > 1 and answer == strs[len(strs) - 2]):
                precision_euler += 1
                break
            if (write_into_file and len(strs) > 1):
                output.write(strs[len(strs) - 2] + '\n')

        # get image name
        # imgs, similarity, geted = get_img_name(similar_hash_codes, feature_path, datum.float_data, dimension, top, answer, 'cos')
        # write similar img into file
        # for img in imgs:
        #    strs = img.split('/')
        #    if (len(strs) > 1 and answer == strs[len(strs) - 2]):
        #        precision_cos += 1
        #        break
        #    if (write_into_file and len(strs) > 1):
        #        output.write(strs[len(strs) - 2] + '\n')

        # get image name
        imgs, similarity, geted = get_img_name(similar_hash_codes, feature_path, datum.float_data, dimension, top,
                                               answer, 'man')
        # write similar img into file
        for img in imgs:
            strs = img.split('/')
            if (len(strs) > 1 and answer == strs[len(strs) - 2]):
                precision_man += 1
                break
            if (write_into_file and len(strs) > 1):
                output.write(strs[len(strs) - 2] + '\n')

        if (write_into_file):
            output.close()

        precision += geted
        index += 1
        count -= 1
        if (count <= 0):
            break
        if (index % 100 == 0):
            print('test imgs num: ' + str(index))
            print('accuracy all: ' + str(precision * 1.0 / index))
            print('accuracy euler: ' + str(precision_euler * 1.0 / index))
            print('accuracy cos: ' + str(precision_cos * 1.0 / index))
            print('accuracy man: ' + str(precision_man * 1.0 / index))

    print('test imgs num: ' + str(index))
    print('accuracy all: ' + str(precision * 1.0 / index))
    print('accuracy euler: ' + str(precision_euler * 1.0 / index))
    print('accuracy cos: ' + str(precision_cos * 1.0 / index))
    print('accuracy man: ' + str(precision_man * 1.0 / index))

    # return precision * 1.0 / index, precision_euler * 1.0 / index, precision_cos * 1.0


'''
        plt.figure("result")
        #print('similar imgs:')
        num = len(imgs)
        row = math.ceil(math.sqrt(num + 1.0))
        col = row
        img = Image.open(datum.name)
        plt.subplot(row, col, 1)
        plt.imshow(img)
        for i in range(num):
            #print(imgs[i])
            plt.subplot(row, col, i + 2)
            img = Image.open(imgs[i])
            plt.imshow(img)
        plt.show()


        #if (len(similarity) >0 and similarity[0] > 0.001):
        #    print('src img')
        #    print datum.name
        #    print datum.float_data
        #    print('similary img')
        #    print imgs[0]
        #    print (similarity[0])
        #    for i in similarity:
        #        print(i)
        #    break
'''

#2017-1-19 herongwei
if __name__ == '__main__':
    dimension = 4096

    train_db_name = '/home/hrw/caffe/examples/VGG/train_features_7991_fc7_leveldb'
    train_db = leveldb.LevelDB(train_db_name)

    #train_it = train_db.RangeIter()

    test_db_name = '/home/hrw/caffe/examples/VGG/test_features_1040_fc7_leveldb'
    test_db = leveldb.LevelDB(test_db_name)


    test_it = test_db.RangeIter()

    correct = 0
    #th = 0.4
    wrong = 0
    #reject = 0
    for test_key, test_value in test_it:
        test_datum = caffe_pb2.Datum.FromString(test_value)
        train_it = train_db.RangeIter()

        Max = 0
        cur_label = ""
        cur_name = ""
        for train_key, train_value in train_it:
            train_datum = caffe_pb2.Datum.FromString(train_value)
            similarity = calculate_cos(test_datum.float_data, train_datum.float_data, dimension)
            if Max < similarity:
                Max = similarity
                cur_label = train_datum.img_label
                cur_name = train_datum.name

        #if Max < th:
           # reject += 1
        #else:
        if cur_label == test_datum.img_label:
            correct += 1
            print (Max)
            print ('trainname'+str(cur_name)+' '+str(cur_label))
            print ('testname' + str(test_datum.name) + ' ' + str(test_datum.img_label))
        if cur_label != test_datum.img_label:
            print (Max)
            print ('trainname' + str(cur_name) + ' ' + str(cur_label))
            print ('testname' + str(test_datum.name) + ' ' + str(test_datum.img_label))
            wrong += 1
            
            #if test_datum.img_label == train_datum.img_label:
            	#print (str(similarity))
            	#print ('testname'+str(test_datum.name)+' '+str(test_datum.img_label))
              	#print ('trainname'+str(train_datum.name)+' '+str(train_datum.img_label))

    print (correct)
    print (wrong)
    #print (reject)

