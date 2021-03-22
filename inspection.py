import sys

def data(file):
    with open (file,'r') as f:
        next(f)
        data=list()
        for row in f:
            data.append(row.strip().split('\t'))
    return data


def gini_and_error(d):
    resultcolumn=list()
    resultlist=list()#get the two type of result
    for row in d:
        result=row[-1]
        resultcolumn.append(result)
        if result not in resultlist:
            resultlist.append(result)
    for row in resultlist:
        result0=0
        result1=0
        for row in resultcolumn:
            if row==resultlist[0]:
                result0+=1
            else:
                result1+=1
    gini_impurity=1-(result0/(result0+result1))**2-(result1/(result0+result1))**2
    error=min(result0/(result0+result1),result1/(result0+result1))
    return gini_impurity, error

def writemetrics(metrics_out,gini_impurity,error):
    writefile=open(metrics_out,'w')
    writefile.write('gini_impurity: %f\n'%(gini_impurity))
    writefile.write('error: %f\n'%(error))
    writefile.close()

def main():
    a_input='politicians_train.tsv'
    a_output='politicians_inspect.txt'
    #a_input=sys.argv[1]
    #a_output=sys.argv[2]
    inputdata=data(a_input)#import
    gini_impurity,error = gini_and_error(inputdata)
    writemetrics(a_output,gini_impurity,error)
    
if __name__=='__main__':
    main()