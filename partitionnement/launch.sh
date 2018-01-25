#!/bin/bash
n=0
while [ $n -le 15 ]
do
    python pyuniprot.py uprot$n.txt go$n.txt 
    n=$((n+1))	
done
