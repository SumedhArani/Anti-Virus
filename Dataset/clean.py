def read():
    #return a list of dictonary
    file = open("/Users/Sumedh/Projects/DA/Dataset/Dataset/train.txt", "r")
    input = file.read().split("  -1\n")
    output = []
    for each in input[:-1]:
        #last case is a blank line and hence read till last but one
        attr = each.split()
        case = {'label':attr[0]}
        for i in range(1,532):
            case[i]=0
        for each_attr in attr[1:]:
            attr_no, val = each_attr.split(":")
            attr_no = int(attr_no)
            val = int(val)
            case[attr_no] = val
        output.append(case)
    file.close()
    return output

def write(input):
    fout = open("data_clean.csv",'w+')
    for each_case in input:
        out = ','.join(map(str, (map(lambda a: each_case[a],range(1,532)))))
        out = out + ',' + str(each_case['label'])+'\n'
        fout.write(out)
    fout.close()

def check(input):
    count =0
    for each_case in input:
        if each_case['label']=='+1':
            count = count+1

def make_list(input):
    label = []
    dataset =[]
    for each_case in input:
        data = list(map(int ,map(lambda a: each_case[a],range(1,532))))
        dataset.append(data)
        label.append(int(each_case['label']))
    return dataset, label


def main():
    input = read()
    write(input)
    check(input)
    dataset, label =make_list(input)
    print(dataset, label)

if __name__ == "__main__":
    main()
