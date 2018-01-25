import sys

def regulon(entry_file):
    
    locustoid = {}
    file = open(entry_file,'r')
    print(entry_file)
    for line in file:
        split = line.split('\t')
        if split[2] != None:
            try:
                locustoid[split[0]]=split[2] 
            except:
                pass
                
    empty_keys = [k for k,v in locustoid.iteritems() if not v]
    for k in empty_keys:
        del locustoid[k]

        
    return locustoid

def convert(entry_file,dico, output_file):
    file1 = open(entry_file,'r')
    file2= open(output_file,'w')
    line = file1.readlines()
    test=[]
    test2=[]
    for l in line:
        test2.append(l)
        l = l.rstrip()
        if l in dico:
            file2.write(dico[l] + '\n')
            test.append(dico[l])
    print(len(test))
    print(len(test2))
    

   
            
reg = regulon(sys.argv[1])
convert(sys.argv[2],reg,sys.argv[3])

    
    
    
