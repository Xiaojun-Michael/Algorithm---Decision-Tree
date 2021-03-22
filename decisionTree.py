import sys
import numpy as np

def data(file):
    with open (file,'r') as f:
        next(f)
        data=list()
        for row in f:
            data.append(row.strip().split('\t'))
    data=np.array(data)
    return data

def header(file):
    with open (file,'r') as f:
        header=f.readline().strip().split('\t')
    header=np.array(header)
    return header

def gini_impurity(d):
    resultcolumn=list()
    resultlist=list()#get the two type of result
    for row in d:
        result=row[-1]
        resultcolumn.append(result)
        if result not in resultlist:
            resultlist.append(result)
    for row in resultlist:
        global result0
        global result1
        result0=0
        result1=0
        for row in resultcolumn:
            if row==resultlist[0]:
                result0+=1
            else:
                result1+=1
    gini_impurity=1-(result0/(result0+result1))**2-(result1/(result0+result1))**2
    return gini_impurity

def splitdata(index,data):
    list0=list()
    list1=list()
    attributelist=list(set(data[:,index]))
    for row in data:
        if row[index]==attributelist[0]:
            list0.append(row)
        else:
            list1.append(row)
    list0=np.array(list0)
    list1=np.array(list1)
    return [list0,list1]

def combined_gini(index,data):#get combined gini on split data
    splitdatas=splitdata(index,data)
    gini_index=0
    totalamount=0
    for i in splitdatas:
        totalamount+=len(i)
    for i in splitdatas:
        weight=len(i)/totalamount
        impurity=gini_impurity(i)
        gini_index+=weight*impurity
    return(gini_index)
    
def select_best_split(data):#select the best split
    gini=2
    index=0
    group=None
    alist=list()
    if len(list(set(data[:,-1])))==1:
        return None
    else:
        for attribute in range(len(data[0])-1):
            alist=list()
            alist=list(set(data[:,attribute]))
            splitdatas=splitdata(attribute,data)
            gini_test=combined_gini(attribute,data)
            if gini_test < gini:
                index=attribute
                group=splitdatas
                gini=gini_test
                value=alist
        return {'index':index,'datas':group,'gini':gini,'attribute':alist}

def majority_vote(data):
    d=dict()
    attributelist=list()
    for row in data:
        attribute=row[-1]
        attributelist.append(attribute)
    for i in list(set(attributelist)):
        d[i]=attributelist.count(i)
    label=0
    v=0
    for i in d.keys():
        if d[i]>v:
            v=d[i]
            label=i
    return label
        
        

class Node:
    def __init__(self,value,depth,data):
        self.left=None
        self.right=None
        self.value=value
        self.depth=depth
        self.data=data
        self.leftvalue=None
        self.rightvalue=None
        self.leftresult=None
        self.rightresult=None

def train_tree(node):
    data=node.data
    depth=node.depth
    value=list(set(data[:,node.value]))#Yes/No
    if len(value) !=1:#if the split attribute is all Y or N
        leftdata,rightdata=select_best_split(data)['datas']
        node.leftvalue=leftdata[0][node.value]
        node.rightvalue=rightdata[0][node.value]
        if depth<max_depth and data.shape[1]-1>depth:
            if select_best_split(leftdata) == None:
                node.leftresult=majority_vote(leftdata)
            else: 
                node.left=Node(select_best_split(leftdata)['index'],depth+1,leftdata)
                train_tree(node.left)

        if depth<max_depth and data.shape[1]-1>depth:
            if select_best_split(rightdata) == None:
                node.rightresult= majority_vote(rightdata)
            else:             
                node.right=Node(select_best_split(rightdata)['index'],depth+1,rightdata)
                train_tree(node.right)

        else:
            node.leftresult=majority_vote(leftdata)
            node.rightresult=majority_vote(rightdata)
    else:
        node.leftvalue=value[0]
        node.leftresult=majority_vote(data)
    return node



def predict(row,a):
    index=a.value
    if row[index]==a.leftvalue:
        if a.left==None:
            return a.leftresult
        else:
            predict(row,a.left)
    if row[index]==a.rightvalue:
        if a.right==None:
            return a.rightresult
        else:
            predict(row,a.right)          

def predict(row, a):
    index=a.value
    if row[index] ==a.leftvalue:
        if a.left==None:
            return a.leftresult
        else:
            return predict(row,a.left)
            
    if  row[index] ==a.rightvalue:
        if a.right==None:
            return a.rightresult
        else:
            return predict(row,a.right)

def predictall(data,a):
    l=list()
    for row in data:
        b=predict(row,a)
        l.append(b)
    return l

def writefile(test_out,testlist):
    writefile=open(test_out,'w')
    for i in testlist:
        writefile.write('%s\n'%(i))
    writefile.close()

def writemetrics(metrics_out,error_train,error_test):
    writefile=open(metrics_out,'w')
    writefile.write('error(train): %f\n'%(error_train))
    writefile.write('error(test): %f\n'%(error_test))
    writefile.close()

def errorrate(predict,real):
    mistake=0
    for j in range(len(predict)):
        if predict[j] != real[j]:
            mistake +=1
    error=mistake/len(predict)
    return error

def main():
    global max_depth
    train_input='politicians_train.tsv'
    test_input='politicians_test.tsv'
    max_depth=3
    train_out='pol_3_train.labels'
    test_out='pol_3_test.labels'
    metrics_out='pol_3_metrics.txt'
    #train_input=sys.argv[1]
    #test_input=sys.argv[2]
    #max_depth=int(sys.argv[3])
    #train_out=sys.argv[4]
    #test_out=sys.argv[5]
    #metrics_out=sys.argv[6]

    train_data=data(train_input)
    test_data=data(test_input)
    head=header(train_input)
    root=Node(select_best_split(train_data)['index'],1,train_data)
    tree=train_tree(root)
    predict_train=predictall(train_data,tree)
    writefile(train_out,predict_train)
    predict_test=predictall(test_data,tree)
    writefile(test_out,predict_test)
    
    real_train=list(train_data[:,-1])
    real_test=list(test_data[:,-1])
    error_train=errorrate(predict_train,real_train)
    error_test=errorrate(predict_test,real_test)
    writemetrics(metrics_out,error_train,error_test)
    
if __name__=='__main__':
    main()

    
