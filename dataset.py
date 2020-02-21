import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from IPython.display import display

class Data :       
    def __init__(self, path):
        #this is public members forward declaration
        self.ID=None
        self.Z1=None
        self.Z2=None
        self.A1=None
        self.A2=None
        self.KinVar=None
        self.TypeTheo=None
        self.TypeCurrent=None
        self.TypeExp =None
        
        self.chi2dof=None
        self.chi2tot=None
        self.dataTheo=None
        self.gridInfo=None
        
        self._path = path  
        print("Reading file from  : ", path)
        self.__updateDataInfo()
        self.__updateGridInfo()
        self.__updateDataTheoGrid()
        self.__getChi2Info()
        display(self.dataTheo)
        print("---------------------------------------------------------")
        
    def __updateDataInfo(self, readUntilLine=9):
        with open(self._path) as file_in:
            all_lines = file_in.readlines()[:readUntilLine];
            for line in all_lines:
                self.__extractDict(line)
                
    def __updateGridInfo (self, lineNum=10):
        self.gridInfo = []
        with open(self._path) as file_in :
            line = file_in.readlines()[lineNum]
            info = line.split("\t")
            for i in info :
                self.gridInfo.append(i.strip())
                
    def __updateDataTheoGrid(self, skipLine=11):
        dataTheo = []
        with open(self._path) as file_in:
            all_lines = file_in.readlines()[skipLine:]
            for line in all_lines :
                vec = []
                dataVec = line.split("\t")
                for data in dataVec:
                    vec.append(float(data))
                dataTheo.append(vec)
        dataTheoArr = np.array(dataTheo)
        self.dataTheo = pd.DataFrame(dataTheoArr, columns= self.gridInfo)      
             
            
    def __extractDict (self, line):
        mylist = line.split(":")
        mylist[0]=mylist[0].strip()  
        mylist[1]=mylist[1].strip()
        if mylist[0]=='ID' :
            self.ID = int(mylist[1])
        elif mylist[0]== "TypeExp":
            self.TypeExp = mylist[1]
        elif mylist[0]=="TypeTheo":
            self.TypeTheo = mylist[1]
        elif mylist[0]=="TypeCurrent":
            self.TypeCurrent = mylist[1]
        elif mylist[0]=="A1":
            self.A1 = float(mylist[1])
        elif mylist[0]=="A2":
            self.A2 = float(mylist[1])
        elif mylist[0]=="Z1":
            self.Z1 = float(mylist[1])
        elif mylist[0]=="Z2":
            self.Z2 = float(mylist[1])
        elif mylist[0]=="KinVar":  
            self.KinVar = self.__extractKinvar(mylist[1])
     
            
    def __extractKinvar (self, line):
        myKinvarList = []
        lineCommaSplit = line.split(",")
        for i in lineCommaSplit :
            i=i.strip("[").strip("]").strip()
            myKinvarList.append(i)
        return myKinvarList   
    
    def __getChi2Info (self):
        self.chi2tot = np.sum(np.array (self.dataTheo["chi2Corr"].to_numpy()))
        self.chi2dof = self.chi2tot/self.dataTheo.shape[0]
        
    def __Q2_E_transform(): #virtual function that will be implemented in DataDISNEU
        raise NotImplementedError()
    
    def filterKinVarBy(self, **kwargs) :
        df =self.dataTheo 
        for key, val in kwargs.items():
            isKeyInKinvar = False
            for kinvar in self.KinVar :
                if key == kinvar :
                    isKeyInKinvar = True
            if isKeyInKinvar==False:
                print ("Keyword does not match with the kinetic variables!")
                return            
            df = df[df[key]==val]
        return df
    
    def plotHistogram (self, bins=50, figsize=[10,10]):
        self.dataTheo.hist(bins=bins, figsize=figsize)
        
    def plotDataTheo (self, **kwargs):
        df = self.filterKinVarBy(**kwargs)
        if len(kwargs) != len(self.KinVar)-1 :
            print("Ambiguity in choosing X-axis!")
            return
        for key in self.KinVar :
            listarg=[]
            for key, val in kwargs.items():
                listarg.append(key)        
            for key in self.KinVar:
                if not (key in listarg):
                    x_axis = key
        Xaxis = df[x_axis].to_numpy()
        Yaxis_data = df["data"].to_numpy()
        Yaxis_theo = df["theory"].to_numpy()
        errorBar = df["totErrorUncor"].to_numpy()
        #### plottinng script#
        keylist=[]
        valList=[]
        string=""
        for key, val in kwargs.items():
            keylist.append(key)
            valList.append(val)
            string = string+ f"{key} = {val}     "
            
       
        fig, ax = plt.subplots()
        plt.plot (Xaxis, Yaxis_theo, "-bo", label = "Theory")
        plt.errorbar(Xaxis, Yaxis_data, errorBar, label="data", ecolor="gray",  marker='o', linestyle='dashed', linewidth=2, markersize=5)
        plt.legend(loc="lower left")
        fig.suptitle(string)
        plt.ylabel(self.TypeTheo)
        plt.xlabel(x_axis)  
        ax.grid(True)
        plt.show()
    
    def getKinVarBins (self):
        kinvarbins = {}
        for key in self.KinVar :
            bins=set(list(self.dataTheo[key]))
            kinvarbins[key]=list(bins)
        return kinvarbins   
    
    def getNormalizedHist(self, df, bins=70):
        columns = list(df.columns)
        assert (columns == self.gridInfo)
        for key in self.KinVar :
            keyvalOrig, occurCountOrig = np.unique(self.dataTheo[key].to_numpy(), return_counts=True)
            keyval, occurCount = np.unique(df[key].to_numpy(), return_counts=True)
            keyval = list(keyval)
            keyGrid = list(keyvalOrig) 
            countRatio = []
            for i in range(len(keyGrid)):
                if keyGrid[i] in keyval :
                    index = keyval.index(keyGrid[i])
                    countRatio.append(occurCount[index]/occurCountOrig[i])
                else :
                    countRatio.append(0)

            fig, ax = plt.subplots()
            plt.plot (keyGrid, countRatio, "-o")
            plt.ylabel("Frequency ratio")
            plt.xlabel(key)
            fig.suptitle(f"Normalized histogram for {key}")
            plt.show()
    

class DataDISNEU (Data) :
    def __init__(self, path):
        Data.__init__(self, path)
        self.__Q2_E_transform()
        
    def __Q2_E_transform(self):
        E_series = self.dataTheo["Q2"]/(2*0.938*self.dataTheo["X"]*self.dataTheo["Y"])
        E_series.rename("E", inplace=True)
        for i in range(E_series.shape[0]):
            E_series.iloc[i]=round(E_series.iloc[i])
        self.dataTheo = self.dataTheo.drop(columns= ["Q2"])
        self.dataTheo= pd.concat([E_series, self.dataTheo], axis=1, sort=False)
        self.__updateInfo()
        display(self.dataTheo)
        
    def __updateInfo (self):
        self.KinVar.insert(0, "E")
        self.KinVar.remove("Q2")
        self.gridInfo = list(self.dataTheo.columns)
    
    def plotDataTheoPerBins (self, x_axis="X", numRows=4, numCols = 3, hspace=0.4, wspace=0.4) :
        fixed=self.KinVar.copy()
        fixed.remove(x_axis)
        assert (len(fixed)==2)
        bin1=self.getKinVarBins()[fixed[0]]
        bin2=self.getKinVarBins()[fixed[1]]        
        for each in bin1 :
            fig = plt.figure(figsize=[20,20])
            fig.subplots_adjust(hspace=hspace, wspace=wspace)
            i=1
            for each2 in bin2 : 
                df = self.dataTheo[(self.dataTheo[fixed[0]]==each) & (self.dataTheo[fixed[1]]==each2)]
                if df.empty :
                    pass
                else : 
                    Xaxis = df[x_axis].to_numpy()
                    Yaxis_data = df["data"].to_numpy()
                    Yaxis_theo = df["theory"].to_numpy()
                    errorBar = df["totErrorUncor"].to_numpy()

                    #### plottinng script#
                    keylist=[]
                    valList=[]
                    string= fixed[0]+" = "+str(each)+"\n"+fixed[1]+" = "+str(each2)


                    ax = fig.add_subplot(numRows, numCols, i)
                    ax.plot (Xaxis, Yaxis_theo, "-bo", label = "Theory")
                    ax.errorbar(Xaxis, Yaxis_data, errorBar, label="data", ecolor="gray",  marker='o', linestyle='dashed', linewidth=2, markersize=5)
                    ax.legend(loc="lower left")
                    ax.text(0.85, 0.85, string,  horizontalalignment='center',verticalalignment='center', transform = ax.transAxes)
                    plt.ylabel(self.TypeTheo)
                    plt.xlabel(x_axis) 
                    plt.grid(True)
                    i=i+1
                                                                                 
class Datasets :
    def __init__(self, listPaths, useEKinvarDisneu=True):
        self.datasets = []
        self._paths = listPaths
        
        for path in self._paths :
            data= Data(path)
            if data.TypeExp == "DISNEU" :
                if useEKinvarDisneu :
                    data = DataDISNEU(path)
            self.datasets.append(data)
        
    def filterBy(self, **kwargs):
        myReturn= []
        temp = self.datasets
        for key, val in kwargs.items():
            if key == "ID":
                for data in temp:
                    if data.ID == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            elif key == "Z1" :
                for data in temp:
                    if data.Z1 == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            elif key == "Z2" :
                for data in temp:
                    if data.Z2 == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            elif key == "A1" :
                for data in temp:
                    if data.A1 == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            elif key == "A2" :
                for data in temp:
                    if data.A2 == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            elif key == "TypeExp" :
                for data in temp:
                    if data.TypeExp == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            elif key == "TypeTheo" :
                for data in temp:
                    if data.TypeTheo == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            elif key == "TypeCurrent" :
                for data in temp:
                    if data.TypeCurrent == val:
                        if data in myReturn:
                            pass
                        else :
                            myReturn.append(data)
            else :
                print("key does not match!")
            
            temp = myReturn
            myReturn = []
        
        print("ID datasets found : ", [data.ID for data in temp])            
        return temp
    
    def plotX_Q2_Coverage(self):
        listExp = []
        for data in self.datasets :
            if ("X" in data.KinVar) and ("Q2" in data.KinVar) :
                listExp.append(data)
        print ("Dataset with X and Q2 kinetic variables : ", [data.ID for data in listExp])
        colors = cm.rainbow(np.linspace(0, 1, len(listExp)))
        fig, ax = plt.subplots()
        for data, color in zip(listExp, colors):
            X= []
            Q2= []
            for i in range(data.dataTheo.shape[0]) :
                X.append(data.dataTheo["X"].iloc[i])
                Q2.append(data.dataTheo["Q2"].iloc[i])
                
            fig.suptitle("Kinematic reach")
            ax.scatter(X,Q2, color=color)          
            plt.ylabel("Q2")
            plt.xlabel("x") 
            ax.grid(True)
        ax.grid(True)
        plt.show()
                  
    def plotChi2Dof (self):
        idList=[]
        cdofList=[]
        for data in self.datasets :
            idList.append(data.ID)
            cdofList.append(data.chi2dof)

        sortedIndex = sorted(range(len(idList)), key=lambda k: idList[k])
        idList.sort()
        temp=[]
        for i in range(len(cdofList)) : 
            temp.append(cdofList[sortedIndex[i]])
            idList[i]=str(idList[i])
            
        index = np.arange(len(idList))
        plt.figure(figsize=(35,10))
        plt.bar(idList, temp)
        plt.xlabel("Dataset's ID", fontsize=25)
        plt.ylabel('Chi2/dof', fontsize=20)
        plt.xticks(index, idList, fontsize=20, rotation=90)
        plt.title('Chi2/dof per dataset', fontsize=30 )
        plt.show()
        
    def getNucleiList(self):
        nucleiList = []
        for data in self.datasets :
            if not [data.A1, data.Z1] in nucleiList:
                nucleiList.append([data.A1, data.Z1])
            if not [data.A2, data.Z2] in nucleiList:
                nucleiList.append([data.A2, data.Z2])
        return nucleiList
            
        
    def plotCombineDataTheo(self, **kwargs):
        pass
                   

