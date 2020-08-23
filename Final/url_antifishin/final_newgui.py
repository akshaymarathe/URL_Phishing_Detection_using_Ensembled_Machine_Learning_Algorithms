import pandas as pd
import numpy as np
from pandas import ExcelWriter
from pandas import ExcelFile
import re
from sklearn.metrics import confusion_matrix 
# from sklearn.cross_validation import train_test_split 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import GaussianNB 
from xgboost import XGBClassifier
from sklearn import metrics 
import matplotlib.pyplot as plt
#importing wx files
import wx
#import the newly created GUI file
import gui
from sklearn import svm
import webbrowser
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification,roc_curve
from sklearn import preprocessing



#extract single feature
def extract_feature_usertest(url):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    
    return length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain




#extract single feature
def extract_feature_usertest_stack(url):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    res=[]

    res.append(length_of_url)
    res.append(http_has)
    res.append(suspicious_char)
    res.append(prefix_suffix)
    res.append(dots)
    res.append(slash)
    res.append(phis_term)
    res.append(sub_domain)
    res.append(ip_contain)
    
    return res



#extract testing feature
def extract_feature_test(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output
    
    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain

#extract training feature
def extract_feature_train(url,output):
    
    
    #length of url
    l_url=len(url)
    if(l_url > 54):
        length_of_url = 1
    else:
        length_of_url = 0

    
    #url has http
    if (("http://" in url) or ("https://" in url)):
        http_has = 1
    else:
        http_has = 0

    #url has suspicious char
    if (("@" in url) or ("//" in url)):
        suspicious_char = 1
    else:
        suspicious_char = 0


    #prefix or suffix
    if ("-" in url):
        prefix_suffix = 1
    else:
        prefix_suffix = 0

    #no of dots
    if ("." in url):
        count = len(url.split('.'))-1
        if (count > 5):
            dots = 0
        else:
            dots = 1
    else:
        dots = 0
    
    #no of slash
    if ("/" in url):
        count = len(url.split('/'))-1
        if (count > 5):
            slash = 0
        else:
            slash = 1
    else:
        slash = 0

    #url has phishing terms
    #("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)
    if (("secure" in url) or ("secure" in url) or ("websrc" in url) or ("ebaysapi" in url) or ("signin" in url) or ("banking" in url) or ("confirm" in url) or ("login" in url)):
        phis_term = 1
    else:
        phis_term = 0
    
    #length of subdomain
    it = url.index("//") + 2
    j = url.index(".")
    c = j - it;
    if (c > 5):
        sub_domain = 0
    else:
        sub_domain = 1
    
    #url contains ip address
    if re.match("\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b",url):
        ip_contain = 1
    else:
        ip_contain = 0
    
    #output
    yn = output

   
        
    return yn,length_of_url,http_has,suspicious_char,prefix_suffix,dots,slash,phis_term,sub_domain,ip_contain
#import train data
def importdata_train(): 
    balance_data = pd.read_csv('id3.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
      
    # Printing the dataset shape 
    print ("Dataset Length: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
#import test data
def importdata_test(): 
    balance_data = pd.read_csv('feature_test.csv',sep= ',', header = 1,usecols=range(1,11),encoding='utf-8') 
      
      
    # Printing the dataset shape 
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape) 
      
    # Printing the dataset obseravtions 
    print ("Dataset: ",balance_data.head()) 
    return balance_data 
#split data into train and test
def splitdataset(balance_data): 
  
    # Seperating the target variable 
    X = balance_data.values[:, 1:10]
    Y = balance_data.values[:, 0] 
  
    # Spliting the dataset into train and test 
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100) 
      
    return X, Y
# Function to perform training with entropy. 
def tarin_using_entropy(X_train, y_train): 
  
    # Decision tree with entropy 
    clf_entropy = DecisionTreeClassifier( 
            criterion = "entropy", random_state = 100, 
            max_depth = 2, min_samples_leaf = 10) 
  
    # Performing training 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    #print("Predicted values:") 
    #print(y_pred) 
    return y_pred 
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred))

    return accuracy_score(y_test,y_pred)*100

#roc
def plot_roc_curve(fpr, tpr ):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



#main funcation
def main():
    # excel_file= 'training.xlsx'
    # df=pd.DataFrame(pd.read_excel(excel_file))
    # excel_file_test= 'test1.xlsx'
    # df1=pd.DataFrame(pd.read_excel(excel_file_test))

    # a=[]
    # b=[]
    # a1=[]
    # b1=[]
    # for url in df['url']:
    #     a.append(url)

    # for output in df['phishing']:
    #     b.append(output)

    # for url1 in df1['url']:
    #     a1.append(url1)

    # for output in df1['result']:
    #     b1.append(output)

    excel_file= 'training.xlsx'
    df1=pd.DataFrame(pd.read_excel(excel_file))
    length=(len(df1) / 100)
    length=round(length * 80)


    df=pd.DataFrame(df1[0:length])
    df1=pd.DataFrame(df1[length:])

    a=[]
    b=[]
    a1=[]
    b1=[]
    for url in df['url']:
        a.append(url)

    for output in df['phishing']:
        b.append(output)

    for url1 in df1['url']:
        a1.append(url1)

    for output in df1['phishing']:
        b1.append(output)

    c=[]
    d=[]
    for url1,output1 in zip(a,b):       
        url=url1
        output=output1
        c.append(extract_feature_train(url,output))

    for url1,output1 in zip(a1,b1):           
        url=url1
        output=output1
        d.append(extract_feature_test(url,output))



    df=pd.DataFrame(c,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df.to_csv('id3.csv', sep=',', encoding='utf-8')

    df_test=pd.DataFrame(d,columns=['r','length_of_url','http_has','suspicious_char','prefix_suffix','dots','slash','phis_term','sub_domain','ip_contain'])

    df_test.to_csv('feature_test.csv', sep=',', encoding='utf-8')  
    
    data_train=importdata_train()
    data_test=importdata_test()
    X, Y = splitdataset(data_train) 
    X1, Y1 = splitdataset(data_test)  
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)
    
    model=XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0,min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
    model.fit(X,Y)
    
    gnb = GaussianNB() 
    gnb.fit(X, Y)
    
    




#STACKING

    df = pd.read_csv("stack.csv")
    df1 = pd.read_csv("feature_test_stack1.csv")
    target = 'r'

    X_train=df.loc[:, df.columns!=target]
    Y_train=df.loc[:, df.columns==target]
    X_test=df1.loc[:, df1.columns!=target]
    Y_test=df1.loc[:, df1.columns==target]

    #Stacking Classifier
    # knn1 = KNeighborsClassifier()
    # log_reg1 = LogisticRegression()
    # svm1 = SVC(probability=True)
    print("___________________________Stacking__________________________________________") 
    clf = svm.SVC(kernel='linear')
    rf2=RandomForestClassifier(bootstrap=True,max_depth= 70,max_features= 'auto',min_samples_leaf= 4,min_samples_split= 10,n_estimators= 400)
    xgb=XGBClassifier()
    classifiers=[clf,xgb]
    sc = StackingClassifier(classifiers,meta_classifier=rf2)  
    sc.fit(X_train, Y_train)
        
    print("_____________Report___________________")
    y_pred4 = sc.predict(X_test) #prediction or testing 
    acc4=cal_accuracy(Y_test, y_pred4)

    #confusion Matrix
    matrix =confusion_matrix(Y_test, y_pred4)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    #fig = pyplot.gcf()
    fig.canvas.set_window_title('Stacking')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    #ROC_AUC curve
    probs = sc.predict_proba(X_test) 
    probs = probs[:, 1]  
    auc = roc_auc_score(Y_test, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(Y_test)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    #fig.canvas.set_window_title('Stacking')
    title='Stacking'
    plot_roc_curve(fpr, tpr)


    #Classification Report
    target_names = ['Yes', 'No']
    prediction=sc.predict(X_test)
    print(classification_report(Y_test, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer = ClassificationReport(sc, classes=classes, support=True)
    visualizer.fit(X_train, Y_train)  
    visualizer.score(X_test, Y_test)
    #fig.canvas.set_window_title('Stacking')  
    g = visualizer.poof()




    print("___________________________XGBOOST__________________________________________") 
    model=XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0,min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)
    model.fit(X,Y)
    y_pred1 = model.predict(X1)
    print("_____________Report___________________")
    acc1=cal_accuracy(Y1, y_pred1)
    # print("_____________user input ___________________")
    
    #confusion Matrix
    import matplotlib.pyplot as plt1
    matrix =confusion_matrix(Y1, y_pred1)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt1.xticks(tick_marks, class_names)
    plt1.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt1.tight_layout()
    plt1.title('Confusion matrix', y=1.1)
    plt1.ylabel('Actual label')
    plt1.xlabel('Predicted label')
    fig.canvas.set_window_title('XGBoost')
    plt.show()

    #ROC_AUC curve
    probs = model.predict_proba(X1) 
    probs = probs[:, 1]  
    auc = roc_auc_score(Y1, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(Y1)
    fpr1, tpr1, thresholds = roc_curve(y_test1, probs)
    #fig.canvas.set_window_title('XGBoost')
    plot_roc_curve(fpr1, tpr1)


    #Classification Report
    target_names = ['Yes', 'No']
    prediction=model.predict(X1)
    print(classification_report(Y1, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer1 = ClassificationReport(model, classes=classes, support=True)
    visualizer1.fit(X, Y)  
    visualizer1.score(X1, Y1)
    #fig.canvas.set_window_title('XGBoost')  
    g = visualizer1.poof()




    print("___________________________SVM__________________________________________") 
    clf = svm.SVC(kernel='linear',probability=True)
    clf.fit(X, Y)
    print("_____________Report___________________")
    y_pred = clf.predict(X1)
    #print(cal_accuracy(Y1, y_pred))
    acc2=cal_accuracy(Y1, y_pred)
    #print("_____________user input ___________________")

    #confusion Matrix
    matrix =confusion_matrix(Y1, y_pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    fig.canvas.set_window_title('SVM')
    plt.show()

    #ROC_AUC curve
    probs = clf.predict_proba(X1) 
    probs = probs[:, 1]  
    auc = roc_auc_score(Y1, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(Y1)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    #fig.canvas.set_window_title('SVM')
    plot_roc_curve(fpr, tpr)


    #Classification Report
    target_names = ['Yes', 'No']
    prediction=clf.predict(X1)
    print(classification_report(Y1, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer = ClassificationReport(clf, classes=classes, support=True)
    visualizer.fit(X, Y)  
    visualizer.score(X1, Y1) 
    #fig.canvas.set_window_title('SVM') 
    g = visualizer.poof()



    print("___________________________Naive Bayes__________________________________________") 
    gnb = GaussianNB() 
    gnb.fit(X, Y)
    print("_____________Report___________________")
    y_pred = gnb.predict(X1)
    #print(cal_accuracy(Y1, y_pred))
    acc3=cal_accuracy(Y1, y_pred)
    #print("_____________user input ___________________")
   
    #confusion Matrix
    matrix =confusion_matrix(Y1, y_pred)
    class_names=[0,1] 
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    fig.canvas.set_window_title('NB')
    plt.show()

    #ROC_AUC curve
    probs = gnb.predict_proba(X1) 
    probs = probs[:, 1]  
    auc = roc_auc_score(Y1, probs)  
    print('AUC: %.2f' % auc)
    le = preprocessing.LabelEncoder()
    y_test1=le.fit_transform(Y1)
    fpr, tpr, thresholds = roc_curve(y_test1, probs)
    #fig.canvas.set_window_title('NB')
    plot_roc_curve(fpr, tpr)


    #Classification Report
    target_names = ['Yes', 'No']
    prediction=gnb.predict(X1)
    print(classification_report(Y1, prediction, target_names=target_names))
    classes = ["Yes", "No"]
    visualizer = ClassificationReport(gnb, classes=classes, support=True)
    visualizer.fit(X, Y)  
    visualizer.score(X1, Y1) 
    #fig.canvas.set_window_title('NB') 
    g = visualizer.poof()




    labels = [' XGboost','SVM','NB','Stacking']
    #sizes = [5, neg_per, neu_per]
    sizes = [acc1,acc2,acc3,acc4]
    index = np.arange(len(labels))
    plt.bar(index, sizes)
    plt.xlabel('Algorithm', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(index, labels, fontsize=10, rotation=0)
    plt.title('comparative study')
    plt.show()
    
    #GUI
    class MainFrame ( wx.Frame ):
	
    	def __init__( self, parent ):
    		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 500,300 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
    
    		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
    
    		bSizer3 = wx.BoxSizer( wx.VERTICAL )
    
    		self.m_staticText2 = wx.StaticText( self, wx.ID_ANY, u"Enter URL", wx.DefaultPosition, wx.DefaultSize, 0 )
    		self.m_staticText2.Wrap( -1 )
    		bSizer3.Add( self.m_staticText2, 0, wx.ALL, 5 )
		
    		self.text1 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    		bSizer3.Add( self.text1, 0, wx.ALL|wx.EXPAND, 5 )
    
    		self.predictButton = wx.Button( self, wx.ID_ANY, u"Predict_XGBOOST", wx.DefaultPosition, wx.DefaultSize, 0 )
    		bSizer3.Add( self.predictButton, 0, wx.ALL|wx.EXPAND, 5 )
		
    		self.m_button2 = wx.Button( self, wx.ID_ANY, u"Predict_SVM", wx.DefaultPosition, wx.DefaultSize, 0 )
    		bSizer3.Add( self.m_button2, 0, wx.ALL|wx.EXPAND, 5 )
    
    		self.m_button3 = wx.Button( self, wx.ID_ANY, u"Predict_NB", wx.DefaultPosition, wx.DefaultSize, 0 )
    		bSizer3.Add( self.m_button3, 0, wx.ALL|wx.EXPAND, 5 )
		
    		self.m_button4 = wx.Button( self, wx.ID_ANY, u"Predict_STACKING", wx.DefaultPosition, wx.DefaultSize, 0 )
    		bSizer3.Add( self.m_button4, 0, wx.ALL|wx.EXPAND, 5 )
		
    		# self.label1 = wx.StaticText( self, wx.ID_ANY, u"Result", wx.DefaultPosition, wx.DefaultSize, 0 )
    		# self.label1.Wrap( -1 )
    		# bSizer3.Add( self.label1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
    		# self.text2 = wx.TextCtrl( self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
    		# bSizer3.Add( self.text2, 0, wx.RIGHT|wx.EXPAND, 5 )
		
		
    		self.SetSizer( bSizer3 )
    		self.Layout()
		
    		self.Centre( wx.BOTH )
		
    		# Connect Events
    		self.predictButton.Bind( wx.EVT_BUTTON, self.click )
    		self.m_button2.Bind( wx.EVT_BUTTON, self.svm )
    		self.m_button3.Bind( wx.EVT_BUTTON, self.nb )
    		self.m_button4.Bind( wx.EVT_BUTTON, self.stacking )
	
    	def __del__( self ):
    		pass
	
	
    	# Virtual event handlers, overide them in your derived class



        #XGBOOST
    	def click( self, event ):
    	    try:
    	        url = self.text1.GetValue()
    	        e=np.array([extract_feature_usertest(url)])
    	        userpredict1 = model.predict(e.reshape(1,-1)) 
    	        if(userpredict1[0]=='no'):
    	            # self.text2.SetValue(str("Legitimate"))
    	            print('Legitimate')
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"POP-UP" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"LEGITIMATE", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )

    	            app3 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            webbrowser.open(url)
    	            app3.MainLoop()

    	        else:
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 200,150), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Error" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"PHISING", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )


    	
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )

    	                	def __del__( self ):
    	                		pass
	
	
	# Virtual event handlers, overide them in your derived class
    	                	def click( self, event ):
    	                		event.Skip()
    	            app2 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            app2.MainLoop() 

    	            # self.text2.SetValue(str("Phising"))
    	            # print('Phising')
    	    except Exception:
    	        print ('error')



        #SVM
    	def svm( self, event ):
    	    clf = svm.SVC(kernel='linear',probability=True)
    	    clf.fit(X, Y)
    	    try:
    	        url = self.text1.GetValue()
    	        e=np.array([extract_feature_usertest(url)])
    	        userpredict1 = model.predict(e.reshape(1,-1)) 
    	        if(userpredict1[0]=='no'):
    	            # self.text2.SetValue(str("Legitimate"))
    	            print('Legitimate')
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"POP-UP" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"LEGITIMATE", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )
    	            app2 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            webbrowser.open(url)
    	            app2.MainLoop()
    	            webbrowser.open(url)

    	        else:
                    
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Error" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"PHISING", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )
    	            app2 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            app2.MainLoop() 

	
    	            def __del__( self ):
    	            	pass
    	            # self.text2.SetValue(str("Phising"))
    	            # print('Phising')
    	    except Exception:
    	        print ('error')



        #NAIVE BAYES
    	def nb( self, event ):
    	    try:
    	        url = self.text1.GetValue()
    	        e=np.array([extract_feature_usertest(url)])
    	        userpredict1 = gnb.predict(e.reshape(1,-1)) 
    	        if(userpredict1[0]=='no'):
    	            # self.text2.SetValue(str("Legitimate"))
    	            print('Legitimate')
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"POP-UP" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"LEGITIMATE", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )
    	            app2 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            webbrowser.open(url)
    	            app2.MainLoop()
    	            webbrowser.open(url)

    	        else:
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Error" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"PHISING", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )
    	            app2 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            app2.MainLoop() 
    	            # self.text2.SetValue(str("Phising"))
    	            # print('Phising')
    	    except Exception:
    	        print ('error')


        #STACKING
    	def stacking( self, event ):
    	    df = pd.read_csv("stack.csv")
    	    df1 = pd.read_csv("feature_test_stack1.csv")
    	    target = 'r'

    	    X_train=df.loc[:, df.columns!=target]
    	    Y_train=df.loc[:, df.columns==target]
    	    X_test=df1.loc[:, df1.columns!=target]
    	    Y_test=df1.loc[:, df1.columns==target]
    	    global url4
    	    url4 = self.text1.GetValue()
    	    xgb=XGBClassifier()
    	    clf = svm.SVC(kernel='linear')
    	    classifiers=[clf,xgb]
    	    rf2=RandomForestClassifier(bootstrap=True,max_depth= 70,max_features= 'auto',min_samples_leaf= 4,min_samples_split= 10,n_estimators= 400)
    	    sc = StackingClassifier(classifiers,meta_classifier=rf2)  
    	    sc.fit(X_train, Y_train)
    	    # e1=extract_feature_usertest(url4)
    	    # userpredict4 = sc.predict(e1)
                
    	    col = df.columns
    	    col=col[:-1]
    	    e4=extract_feature_usertest_stack(url4)
    	    output_data=e4
    	    output_data=pd.DataFrame([output_data],columns = col)
    	    userpredict4 = sc.predict(output_data)
    	    if(userpredict4[0]==0):
    	        # self.text2.SetValue(str("Legitimate"))
    	        print('Legitimate')
    	        class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"POP-UP" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"LEGITIMATE", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )
    	        app2 = wx.App(False)
    	        frame = MyDialog1(None)
    	        frame.Show(True)
    	        webbrowser.open(url)
    	        app2.MainLoop()

    	    else:
    	            class MyDialog1 ( wx.Dialog ):
	
    	                def __init__( self, parent ):
    	                	wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 159,114 ), style = wx.DEFAULT_DIALOG_STYLE )
		
    	                	self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
    	                	sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Error" ), wx.VERTICAL )
		
    	                	self.m_staticText1 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"PHISING", wx.DefaultPosition, wx.DefaultSize, 0 )
    	                	self.m_staticText1.Wrap( -1 )
    	                	sbSizer1.Add( self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL, 5 )
		
		
    	                	self.SetSizer( sbSizer1 )
    	                	self.Layout()
		
    	                	self.Centre( wx.BOTH )
    	            app2 = wx.App(False)
    	            frame = MyDialog1(None)
    	            frame.Show(True)
    	            app2.MainLoop() 
    	            # self.text2.SetValue(str("Phising")) 
    	            # print('Phising')
    app1 = wx.App(False)
    
    
    frame = MainFrame(None)
    frame.Show(True)
    app1.MainLoop() 


  
if __name__== "__main__":
  main()





    
