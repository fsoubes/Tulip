# Powered by Python 3.6

# To cancel the modifications performed by the script
# on the current graph, click on the undo button.

# Some useful keyboards shortcuts : 
#   * Ctrl + D : comment selected lines.
#   * Ctrl + Shift + D  : uncomment selected lines.
#   * Ctrl + I : indent selected lines.
#   * Ctrl + Shift + I  : unindent selected lines.
#   * Ctrl + Return  : run script.
#   * Ctrl + F  : find selected text.
#   * Ctrl + R  : replace selected text.
#   * Ctrl + Space  : show auto-completion dialog.

from tulip import *
from tulipgui import *
from math import *
import numpy as np
from scipy.stats.stats import pearsonr
from scipy import *
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from scipy.stats import *
import scipy.stats as stats
import warnings

import csv

#######################
###### PART 1 #########
#######################

def cloned(gr):
	
	"""
	add a clone and a copy of a graph if not already created
	"""
	
	if gr.getSubGraph("Clone") is None:
		clone = gr.addCloneSubGraph("Clone")
	else :
		clone =gr.getSubGraph("Clone")
	if gr.getSubGraph("Copy") is None:
		copy = gr.addCloneSubGraph("Copy")
	else :
		copy =gr.getSubGraph("Copy")
		
	return copy,clone


def displaylabels(gr):
	
	"""
	add small details for the labels and nodes
	"""
	
	viewBorderColor = gr.getColorProperty("viewBorderColor")
	viewShape = gr.getIntegerProperty("viewShape")
	viewLabelBorderWidth = gr.getDoubleProperty("viewLabelBorderWidth")
	param = tlp.LabelPosition.Center
	viewLabelPosition = gr.getIntegerProperty("viewLabelPosition")
	Locus = gr.getStringProperty("Locus")
	viewLabel = gr.getStringProperty("viewLabel")
	for n in gr.getNodes():
			viewBorderColor[n] = tlp.Color.Red
			viewShape[n] = tlp.NodeShape.Cube
			viewLabel[n] =  str(Locus[n])
			viewLabelPosition[n] = param
			viewLabelBorderWidth[n] = 1.00

def nodesize(gr):
	
	""" 
	modify the node's size
	"""
	
	viewBorderWidth = gr.getDoubleProperty("viewBorderWidth")
	viewSize = gr.getSizeProperty("viewSize")	
	baseSize = tlp.Size(1500,500,1)
	for n in gr.getNodes():
		viewSize[n] = baseSize  
		viewBorderWidth[n] = 1.00


	
#	return shap


def changeedge(gr):
	
	"""
	modify color/shape of the edges in function of the regulation
	"""
	
	viewTgtAnchorShape =  gr.getIntegerProperty("viewTgtAnchorShape")
	viewTgtAnchorSize =  gr.getSizeProperty("viewTgtAnchorSize")
	Negative = gr.getBooleanProperty("Negative")
	Positive = gr.getBooleanProperty("Positive")
	viewColor = gr.getColorProperty("viewColor")
	viewShape = gr.getIntegerProperty("viewShape")
	
	negandnotpos = tlp.EdgeExtremityShape.Circle
	notnegandnotpos = tlp.EdgeExtremityShape.Diamond
	notnegandpos = tlp.EdgeExtremityShape.Star
	
	blue = tlp.Color.Blue
	gold = tlp.Color.Red
	green = tlp.Color(0,150,0)
	purple = tlp.Color.Purple
	
	cpt = cptt = cpttt = cptttt = 0

	for e in gr.getEdges():		
		if Negative[e] and Positive[e]:
			cptt += 1
			viewColor[e] = purple
			
		if Negative[e] and not Positive[e]:
			cpttt += 1
			viewColor[e] = blue
			viewTgtAnchorShape[e] = negandnotpos
			
		if not Negative[e] and Positive[e]:
			cpt += 1
			viewTgtAnchorShape[e] = notnegandpos
			viewColor[e] = green
			
		if not Negative[e] and not Positive[e]:
			cptttt += 1
			viewColor[e] = gold 
			viewTgtAnchorShape[e] = notnegandnotpos
				
	print("we obtained the following regulation Negative and Positive: {} Negative and not Positive: {} not Negative and Positive: {} and not Negative and Positive: {}".format(cptt, cpttt, cpt, cptttt))
	

def edge_bundling(gr):
	"""
	bundles edges for a better representation of the graph
	"""
	params = tlp.getDefaultPluginParameters('Edge bundling', gr)
	success = gr.applyAlgorithm('Edge bundling', params)



def dessinerModeleForce(gr):
	
	"""
	Apply FM^3 for better disposal of our nodes
	"""
	
	layout = gr.getLayoutProperty("viewLayout")
	params = tlp.getDefaultPluginParameters('FM^3 (OGDF)', gr)
	params['Unit edge length'] = 2
	success = gr.applyLayoutAlgorithm('FM^3 (OGDF)', layout, params)
	centerViews = True


def AddView(gr):
	
	"""
	CreateView with labels out of the nodes
	Central Perspective
	"""
	
	nodeLinkView = tlpgui.createView("Node Link Diagram view",gr,{},True)
	tlpgui.getOpenedViewsWithName("Node Link Diagram view")
	renderingParameters = nodeLinkView.getRenderingParameters()
	renderingParameters.setLabelScaled(False) #true fit, false dont fit
	renderingParameters.getDisplayFilteringProperty()
	#print(renderingParameters.isLabelScaled())## False labels are in the nodes and are not rly visible!
	nodeLinkView.setRenderingParameters(renderingParameters) 
	nodeLinkView.centerView()	
	nodeLinkView.draw()
	updateVisualization(centerViews = True)
	return nodeLinkView

def savescren(nodeLinkView,name):
	
	"""
	Save png for report
	"""
	
	nodeLinkView.saveSnapshot("C:/Users/franck1337/Desktop/",name,".png",1024 , 768)


#######################
###### PART 2 #########
##### Partitionnement #
#######################


def isnormal(gr,tps):
	
	tps_all = []
	for i in range(0,len(tps)):
		for n in gr.getNodes():
			tps_all.append(tps[i][n])
	norm = normaltest(tps_all)
	h = sorted(tps_all)
	h = np.trim_zeros(h)
	fit = stats.norm.pdf(h,np.mean(h),np.std(h))

	plt.plot(h,fit,'-o')
	plt.xlabel('Niveaux expression')
	plt.ylabel('Frequence')
	pl.hist(h,normed=True)
	pl.show()
	

def delnode(nodesource,tp_s) :
	
	"""
	return the euclidean distance of two points
	"""
	
	somme = 0
	for j in range (len(tp_s)) :
		somme += tp_s[j][nodesource]

	return somme
	
def pearson(nodesource,nodetarget, tp_s) :
	
	"""
	return the p-value of two set of points
	"""
	
	pearsonn=[]
	pearsonnext=[]
	for j in range (len(tp_s)) :
		pearsonn.append(tp_s[j][nodesource])
		pearsonnext.append(tp_s[j][nodetarget])	
	warnings.filterwarnings('error')
	try:
		R = pearsonr(pearsonn, pearsonnext)
		d = 1-R[0]
		return d,R[1]	
	except:
		pass
		return 0,1
		

	
	
def construct(gr,tp):
	
	if gr.getSubGraph("Partitionnement") is None:
		partitionnement = gr.addCloneSubGraph("Partitionnement")
	else :
		partitionnement =gr.getSubGraph("Partitionnement")	
		
	poids = partitionnement.getDoubleProperty("Poids")
	pvalue = partitionnement.getDoubleProperty("p_value")	
	
	for e in partitionnement.getEdges():
		partitionnement.delEdge(e)
		
	for n in partitionnement.getNodes():
		if delnode(n,tp) == 0:
			partitionnement.delNode(n)
				
	partitionning(partitionnement, poids,pvalue, tp)
	return partitionnement


def clustering(gr):
	
	if gr.getSubGraph("Clustering") is None:
		clustering = gr.addCloneSubGraph("Clustering")
	else :
		clustering =gr.getSubGraph("Clustering")
		
	params = tlp.getDefaultPluginParameters('MCL Clustering')
	#params['inflate'] = 
	#params['pruning'] = ...
	params['poids'] = gr.getDoubleProperty("Poids")
	resultClustering = gr.getDoubleProperty('resultMetric')
	success = gr.applyDoubleAlgorithm('MCL Clustering', resultClustering, params)
	#params = tlp.getDefaultPluginParameters('Equal Value', gr)
	#params['Property'] = resultClustering
	#success = graph.applyAlgorithm('Equal Value', params)
	
	return clustering



def partitionning(gr,poids,pvalue,tp):
	
	for n in gr.getNodes():
		for nextn in gr.getNodes():
			d = pearson(n,nextn, tp)
			if n == nextn:
				continue
			d,pval= pearson(n,nextn,tp)
			e = gr.existEdge(n,nextn,False)
			if e.isValid() == False and pval < 0.01 and d != 0 and d<0.10 :
				edge = gr.addEdge(n,nextn)
				poids[edge] = d
				pvalue[edge] = pval


#######################
###### PART 3 #########
###### HeatMap ########
#######################

def attributevalues(gr,listetps,Locus):
	
	Locusname= []
	Locus = graph.getStringProperty("Locus")

	for n in gr.getNodes():
		Locusname.append(Locus[n])
	size=gr.numberOfNodes()
	print(size)
	tps_all = []
	for i in range(len(listetps)):
		for n in gr.getNodes():
			tps_all.append(listetps[i][n])
	expr = np.reshape(tps_all,(17,1363))
	return expr,Locusname
	


def construireGrille(gr, lignes, colonnes,liste,listegene,flag,layout):
	
	
	
	if not flag:
		if gr.getSubGraph("Heatmap_no_clust") is None:
			heat = gr.addCloneSubGraph("Heatmap_no_clust")
		else :
			heat =gr.getSubGraph("Clustering")
			
	if flag:
		if gr.getSubGraph("Heatmap_clust") is None:
			heat = gr.addCloneSubGraph("Heatmap_clust")
		else :
			heat =gr.getSubGraph("Clustering")
			
	nodes = {}
	for i in heat.getNodes():
		heat.delNode(i)
		Locus = heat.getStringProperty("Locus")
		gene_exp=heat.getDoubleProperty("Gene_expression")
		
		viewSize = gr.getSizeProperty("viewSize")
		_time_="_time_"
		
	for i in range(0, colonnes):
		nodes[i] = {}
		for j in range(0 , lignes):
			nodes[i][j] = heat.addNode()			
			Locus[nodes[i][j]]=(liste[j]+str(_time_)+str(i+1))		
			if not flag:
				gene_exp[nodes[i][j]]= listegene[i][j]
			else:
				gene_exp[nodes[i][j]]= listegene[j][i]
				
	decalageX = 100
	decalageY = 1.5
	for i in nodes:
		for j in nodes[i]:
			layout[nodes[i][j]] = tlp.Coord(i*decalageX,j*decalageY,0)
		

						
	return heatmap(heat)
			
			

def getordervalues(gr,tp_s):

	cpt=0
	for i in gr.getNodes():
		cpt+=1
	rMetric = gr.getDoubleProperty("resultMetric")
	max_metric = rMetric.getNodeDoubleMax()
	Locus = gr.getStringProperty("Locus")
	m =0
	finallocus=[]
	finaltp = []
	while m <= max_metric :
		for i in gr.getNodes():
			if rMetric[i] == m :
				finallocus.append(Locus[i])
				for p in range(len(tp_s)):
					finaltp.append(tp_s[p][i])
							
		m+=1
		
	
	lentp_s=len(tp_s)
	tps2d = np.reshape(finaltp,(cpt,lentp_s))
	
	
	return tps2d,finallocus
					

def heatmap(heat):
	
	viewShape = heat.getIntegerProperty("viewShape")
	gene_exp=heat.getDoubleProperty("Gene_expression")
	viewSize = heat.getSizeProperty('viewSize')
	max_exp = gene_exp.getNodeDoubleMax()
	min_exp = gene_exp.getNodeDoubleMin()
	viewBorderColor = heat.getColorProperty("viewBorderColor")
	viewColor = heat.getColorProperty("viewColor")
	colorScale = tlp.ColorScale([])
	colors = [tlp.Color.Green, tlp.Color.Black,tlp.Color.Red]
	colorScale.setColorScale(colors)
	
	for n in heat.getNodes():
		viewColor[n]=colorScale.getColorAtPos((gene_exp[n]-min_exp)/(max_exp-min_exp))
		viewBorderColor[n]=viewColor[n]
		viewSize[n]=tlp.Size(100,1.5,1)
	
	return heat
		
		

#######################
###### ANALYSE ########
#######################

def getcluster(gr):
	
	rMetric = gr.getDoubleProperty("resultMetric")
	Locus = gr.getStringProperty("Locus")
	m=0
	max_metric = rMetric.getNodeDoubleMax()
	#finallocus = []
	#finallocus.append("Cluster 0")
	while m <= 15 :
		finallocus=[]
		for i in gr.getNodes():
			if rMetric[i] == m :
				finallocus.append(str(Locus[i]))
		fl = open("cluster" + str(m) + ".txt", "w")
		for gene in finallocus :
			fl.write(gene+ "\n")
		fl.close()
		m+=1

		
	return finallocus
  	

def Regulon(gr):
	Listeregul=[]
	Locus = gr.getStringProperty("Locus")
	for i in gr.getNodes():
		if gr.deg(i) > 40:
			Listeregul.append(Locus[i])
	return Listeregul


#######################
###### MAIN ###########
#######################

def main(graph): 
	

## Clone and Copy

  copy,clone = cloned(graph)
    
## Attributes

  Locus = graph.getStringProperty("Locus")
  Negative = graph.getBooleanProperty("Negative")
  Positive = graph.getBooleanProperty("Positive")
  tp1_s = graph.getDoubleProperty("tp1 s")
  tp2_s = copy.getDoubleProperty("tp2 s")
  tp3_s = graph.getDoubleProperty("tp3 s")
  tp4_s = graph.getDoubleProperty("tp4 s")
  tp5_s = graph.getDoubleProperty("tp5 s")
  tp6_s = graph.getDoubleProperty("tp6 s")
  tp7_s = graph.getDoubleProperty("tp7 s")
  tp8_s = graph.getDoubleProperty("tp8 s")
  tp9_s = graph.getDoubleProperty("tp9 s")
  tp10_s = graph.getDoubleProperty("tp10 s")
  tp11_s = graph.getDoubleProperty("tp11 s")
  tp12_s = graph.getDoubleProperty("tp12 s")
  tp13_s = graph.getDoubleProperty("tp13 s")
  tp14_s = graph.getDoubleProperty("tp14 s")
  tp15_s = graph.getDoubleProperty("tp15 s")
  tp16_s = graph.getDoubleProperty("tp16 s")
  tp17_s = graph.getDoubleProperty("tp17 s")
  tp_s = [ tp1_s , tp2_s ,tp3_s ,tp4_s ,tp5_s ,tp6_s ,tp7_s ,tp8_s ,tp9_s ,tp10_s ,tp11_s ,tp12_s ,tp13_s,tp14_s,tp15_s,tp16_s,tp17_s]
  viewBorderColor = graph.getColorProperty("viewBorderColor")
  viewBorderWidth = graph.getDoubleProperty("viewBorderWidth")
  viewColor = graph.getColorProperty("viewColor")
  viewFont = graph.getStringProperty("viewFont")
  viewFontSize = graph.getIntegerProperty("viewFontSize")
  viewIcon = graph.getStringProperty("viewIcon")
  viewLabel = graph.getStringProperty("viewLabel")
  viewLabelBorderColor = graph.getColorProperty("viewLabelBorderColor")
  viewLabelBorderWidth = graph.getDoubleProperty("viewLabelBorderWidth")
  viewLabelColor = graph.getColorProperty("viewLabelColor")
  viewLabelPosition = graph.getIntegerProperty("viewLabelPosition")
  viewLayout = graph.getLayoutProperty("viewLayout")
  viewMetric = graph.getDoubleProperty("viewMetric")
  viewRotation = graph.getDoubleProperty("viewRotation")
  viewSelection = graph.getBooleanProperty("viewSelection")
  viewShape = graph.getIntegerProperty("viewShape")
  viewSize = graph.getSizeProperty("viewSize")
  viewSrcAnchorShape = graph.getIntegerProperty("viewSrcAnchorShape")
  viewSrcAnchorSize = graph.getSizeProperty("viewSrcAnchorSize")
  viewTexture = graph.getStringProperty("viewTexture")
  viewTgtAnchorShape = graph.getIntegerProperty("viewTgtAnchorShape")
  viewTgtAnchorSize = graph.getSizeProperty("viewTgtAnchorSize")

## graph information & params
	
  numberedges = clone.numberOfEdges()
  numbernodes = clone.numberOfNodes()
  numberexpre = len(tp_s)
  decalageX = 100
  decalageY = 1.5


 
##Première partie
	
  displaylabels(copy)
  nodesize(copy)
  changeedge(copy)
  dessinerModeleForce(copy) 
  AddView(copy)
  edge_bundling(copy)
  #savescren(notscaled(copy))

##Seconde partie

  
  #partit = construct(copy,tp_s)
  #dessinerModeleForce(copy) 
  #clustergr = clustering(partit) 
  #numnodefiltered = clustergr.numberOfNodes()
  #isnormal(graph,tp_s)

##Troisème partie 

 

#heatmap without clustering

  #locusheatv1,listv1 = attributevalues(copy,tp_s,Locus)
  #heatmpv1 = construireGrille(copy, numbernodes,numberexpre ,listv1,locusheatv1,False,viewLayout)

  
#heatmap with clustering

  #locusheatv2, listv2 = getordervalues(clustergr,tp_s)
  #heatmp = construireGrille(clustergr, numnodefiltered, numberexpre, listv2,locusheatv2,True,viewLayout	)



  ## Analyse
  
  parti = copy.getSubGraph("Partitionnement")
  
  listecluster = getcluster(parti)
  #locusheatv2, listv2 = getordervalues(parti,tp_s)
##Analyse

   	

  






