#!/usr/bin/env python
import sys
from bioservices import UniProt


def obtained(liste,output_file):

    uprot=[]
    u = UniProt()
    for ids in liste:
        sp = " "
        name= "Escherichia coli (strain K12)"
        final = str(ids)+sp+name
        d = u.quick_search(final, limit=1)
        for entry in d.keys():
            uprot.append(entry)

    
    
    dd = u.get_df(uprot)
    GO = dd["Gene ontology (GO)"]
    
    file2= open(output_file,'w')

    file2.write("numero uniprot est le suivant \n \n")
    for i in range(len(liste)):
        file2.write(str(liste[i] +" "))

    file2.write("\n")
    file2.write("\n")
    file2.write("ID accession uniprot est le suivant \n")
    for i in range(len(uprot)):
        file2.write(str(uprot[i] + "  "))

    file2.write("\n")
    file2.write("\n")
    file2.write("L'enrichissement ontologique pour ces termes est la suivante \n\n")
    for i in range(len(GO)):
        for j in range(len(GO[i])):
            file2.write(GO[i][j] )
        file2.write("\n \n")
    
            
    return GO

def getid(entry_file):
    file1 = open(entry_file,'r')
    line = file1.readlines()
    listid=[]
    for l in line:
        l = l.rstrip()
        listid.append(l)
    print(len(listid))
    return (listid)



reg = getid(sys.argv[1])
res = obtained(reg,sys.argv[2])

