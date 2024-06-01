import tkinter
import tkinter.messagebox
import sqlite3
from tkinter import * 
import tkinter as tk
from random import *
import string
#import sqlite3

entry_1 = None;
entry_2 = None;
entry_3 = None;

class ForFrames(tk.Tk):
    
     def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)  
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)


        self.frames = {}
        for F in (Registerform,Login):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame


            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("Registerform")

     def show_frame(self, page_name):

        frame = self.frames[page_name]
        frame.tkraise() 

class Registerform(tk.Frame):
    def __init__(self,parent,controller):

        tk.Frame.__init__(self, parent)
        self.controller = controller




            # convert registered userinfo to json file
        def regPress():
            usern = entry_1.get()
            passw = entry_2.get()
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            if entry_2.get() == entry_3.get() and not len(entry_1.get()) == 0:
                c.execute("CREATE TABLE IF NOT EXISTS 'entries' (username TEXT, password TEXT)")
                c.execute("INSERT INTO entries(username,password)VALUES(?,?)",(usern,passw))
                MsgBox = tkinter.messagebox.showinfo("Success","Registered, click OK to login")
                if MsgBox == 'ok':
                    controller.show_frame("Login")
            conn.commit()
            
            if entry_2.get() != entry_3.get():
                     tkinter.messagebox.showinfo("Failed","Passwords don't match")
            elif len(entry_1.get()) == 0:
                    tkinter.messagebox.showinfo("Failed","Please enter a username")

     

                
        registerframe1 = Frame(self)
        registerframe1.pack(fill=X)

        registerframe2 = Frame(self)
        registerframe2.pack(fill=X)

        registerframe3 = Frame(self)
        registerframe3.pack(fill=X)

        registerframe6 = Frame(self)
        registerframe6.pack(fill=X)

        label_1 = tk.Label(registerframe1, text="Username")
        label_2 = tk.Label(registerframe2, text="Password")
        label_3 = tk.Label(registerframe3, text="Password confirmation")
        

        label_1.pack(side=LEFT,padx=5,pady=5)
        label_2.pack(side=LEFT,padx=5,pady=5)
        label_3.pack(side=LEFT,padx=5,pady=5)
        

        entry_1 = Entry(registerframe1, width=50)
        entry_2 = Entry(registerframe2, width=50, show='*')
        entry_3 = Entry(registerframe3, width=50, show='*')

        entry_1.pack(side=RIGHT,padx=100)
        entry_2.pack(side=RIGHT,padx=100)
        entry_3.pack(side=RIGHT,padx=100)
        
        

         

        def randompw():
            
                characters = string.ascii_letters + string.digits
                pwmessage = "".join(choice(characters) for x in range (randint(8, 12)))
                print (pwmessage)

                

                registerframePW = Frame(self)
                registerframePW.pack(fill=X)

                label_PW = tk.Label(registerframePW, text="This is your password. Please, never share it !")
                label_PW.pack()

                entryText = tk.StringVar()
                entry_PW = Entry(registerframePW, width=50, textvariable=entryText)
                entryText.set(pwmessage)
                entry_PW.pack()

            
            
            
            
              
        #### nupud



        button1 = tk.Button(self, text="Register", command=regPress)
        button2 = tk.Button(self, text="Already have an account? Login",command=lambda: controller.show_frame("Login"))
        button3 = tk.Button(self, text="Create a random password",command=randompw)

        
        button2.pack(side=BOTTOM)
        button1.pack(side=TOP,padx=5,pady=5)
        button3.pack(side=BOTTOM)
         


        

class Login(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        self.controller = controller

        #database
        def LogPress():
            usern = entry_1.get()
            passw = entry_2.get()
            if usern == '' or passw == '':
                tkinter.messagebox.showinfo("Failed","Please enter username and password")

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("SELECT * FROM entries WHERE username = ? and password = ?",(usern,passw))
            if c.fetchall():
                tkinter.messagebox.showinfo(title = "Successfully logged in", message = "Welcome!!! ")
            else:
                tkinter.messagebox.showerror(title = "Error", message = "incorrect username or password")

            c.close()   


        registerframe4 = Frame(self)
        registerframe4.pack(fill=X)

        registerframe5 = Frame(self)
        registerframe5.pack(fill=X)

        label_1 = tk.Label(registerframe4, text="Username")
        label_2 = tk.Label(registerframe5, text="Password")

        label_1.pack(side=LEFT,padx=5,pady=5)
        label_2.pack(side=LEFT,padx=5,pady=5)

        entry_1 = Entry(registerframe4, width=50)
        entry_2 = Entry(registerframe5, width=50, show='*')

        entry_1.pack(side=RIGHT,padx=100)
        entry_2.pack(side=RIGHT,padx=100)

        button1 = tk.Button(self, text="Login",command=LogPress)
        button1.pack(side=TOP)
        button2 = tk.Button(self, text="Don't have an account?", command=lambda: controller.show_frame("Registerform"))
        button2.pack(side=BOTTOM)

        
       
            

    def close_window(self):
        self.master.destroy()


if __name__ == "__main__":


    app = ForFrames()
    app.geometry("700x250")
    app.mainloop()
    
#IMPORTING LIBRARIES
import datetime
import hashlib
import json
from tinyec import registry
from Crypto.Cipher import AES
import secrets
import hashlib, binascii
import pandas as pd
import numpy as np

import os
  
#CREATING BLOCKCHAIN CLASS 
class Blockchain:

    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash}
        self.chain.append(block)
        return block
        
    def print_previous_block(self):
        return self.chain[-1]
        
    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
          
        while check_proof is False:
            hash_operation = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
                  
        return new_proof

  
    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()
  
    def chain_valid(self, chain):
        previous_block = chain[0]
        block_index = 1
          
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
                
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(
                str(proof**2 - previous_proof**2).encode()).hexdigest()
              
            if hash_operation[:4] != '0000':
                return False
            previous_block = block
            block_index += 1
          
        return True

#ECC ENCRYTION AND DECRYPTION WITH AES
def encryption_AES(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def decryption_AES(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def ecc_to_256_bitkey(point):
    sha = hashlib.sha256(int.to_bytes(point.x, 32, 'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()

curve = registry.get_curve('brainpoolP256r1')


def ECC_Encrytion(msg, pubKey):
    ciphertextPrivKey = secrets.randbelow(curve.field.n)
    sharedECCKey = ciphertextPrivKey * pubKey
    secretKey = ecc_to_256_bitkey(sharedECCKey)
    ciphertext, nonce, authTag = encryption_AES(msg, secretKey)
    ciphertextPubKey = ciphertextPrivKey * curve.g
    return (ciphertext, nonce, authTag, ciphertextPubKey)

def ECC_Decrytion(storedMsg, privKey):
    (ciphertext, nonce, authTag, ciphertextPubKey) = storedMsg
    sharedECCKey = privKey * ciphertextPubKey
    secretKey = ecc_to_256_bitkey(sharedECCKey)
    plaintext = decryption_AES(ciphertext, nonce, authTag, secretKey)
    return plaintext

#-------------------------------------------------------------------------------------------------
blockchain = Blockchain()
previous_block = blockchain.print_previous_block()
previous_proof = previous_block['proof']
proof = blockchain.proof_of_work(previous_proof)
previous_hash = blockchain.hash(previous_block)
block = blockchain.create_block(proof, previous_hash)

#lOADING DATASET
df=pd.read_csv('dataset.csv')
df=df.iloc[:10]


lak = df.to_numpy().flatten()

encrypt = []
decrypt = []
for j in lak:
    j = str(j)
    msg = str.encode(j)    
    privKey = secrets.randbelow(curve.field.n)
    pubKey = privKey * curve.g
    
    encryptedMsg = ECC_Encrytion(msg, pubKey)
    encrypt.append(encryptedMsg)
      
    response = {'message': encryptedMsg,
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash']} 
    response2 = {'chain': blockchain.chain,
                    'length': len(blockchain.chain)} 
    valid = blockchain.chain_valid(blockchain.chain)
          
    if valid:
        print( 'The Blockchain is valid.')
        storedMsg=response["message"]
        #print(storedMsg)
        decryptedMsg = ECC_Decrytion(storedMsg, privKey)
        decryptedMsg = decryptedMsg.decode('utf-8')
        decrypt.append(decryptedMsg)
        
        print("decrypted msg:", decryptedMsg)
    else:
        print( 'The Blockchain is not valid.')
print("please waiting")
    
"Blockchain Encryption and decryption "

def AES_Encryption(msg, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM)
    ciphertext, authTag = aesCipher.encrypt_and_digest(msg)
    return (ciphertext, aesCipher.nonce, authTag)

def AES_Decryption(ciphertext, nonce, authTag, secretKey):
    aesCipher = AES.new(secretKey, AES.MODE_GCM, nonce)
    plaintext = aesCipher.decrypt_and_verify(ciphertext, authTag)
    return plaintext

def ECC_bit_key_generation(point):
    sha = hashlib.sha256(int.to_bytes(point.x, 32, 'big'))
    sha.update(int.to_bytes(point.y, 32, 'big'))
    return sha.digest()

curve = registry.get_curve('brainpoolP256r1')

def ECC_Encryption(msg, pubKey):
    ciphertextPrivKey = secrets.randbelow(curve.field.n)
    sharedECCKey = ciphertextPrivKey * pubKey
    secretKey = ECC_bit_key_generation(sharedECCKey)
    ciphertext, nonce, authTag = AES_Encryption(msg, secretKey)
    ciphertextPubKey = ciphertextPrivKey * curve.g
    return (ciphertext, nonce, authTag, ciphertextPubKey)

def ECC_Decryption(encryptedMsg, privKey):
    (ciphertext, nonce, authTag, ciphertextPubKey) = encryptedMsg
    sharedECCKey = privKey * ciphertextPubKey
    secretKey = ECC_bit_key_generation(sharedECCKey)
    plaintext = AES_Decryption(ciphertext, nonce, authTag, secretKey)
    return plaintext

#----------------------------------------------------------------------------------------


df1 = pd.read_csv("dataset.csv") 
df1=df1.iloc[:10]
df1.shape

column_names = list(df.columns)

result = df.values

print("Encrypting and Decrypting the CSV file...")  
empty = []
empty_decoded = []
for i in result:
    for j in i:
        a = str(j)
        en = a.encode()
        s = ECC_Encrytion(en, pubKey)
        b = binascii.hexlify(s[0])
        encoded_text = b.decode('utf-8')
        empty.append(encoded_text)
        #print(f"Encoded Text : {encoded_text}")
        
        
        de = ECC_Decryption(s, privKey)
        decoded_text = de.decode('utf-8')
        empty_decoded.append(decoded_text)
        #print(f"Decoded Text  : {decoded_text}")
     
encrypted_df = pd.DataFrame(np.array(empty).reshape(10,45),columns = column_names)

print("Encryption Completed and written as encryption.csv file")
encrypted_df.to_csv(r'encrypted.csv',index = False)



print("decryption Completed and written as Decryption.csv file")

decrypted_df = pd.DataFrame(np.array(decrypt).reshape(10,45),columns =df.columns)
decrypted_df.to_csv(r'decryption.csv',index = False)

decrypted_df.head()  
"Import Libaries "

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics




print("==================================================")
print("Block chain in the 5G/6G technology  Dataset")
print(" Process - Block chain in the 5G/6G technology  Attack Detection")
print("==================================================")


##1.data slection---------------------------------------------------
#def main():
dataframe=pd.read_csv("dataset.csv")

print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()
print("please waiting")


 #2.pre processing--------------------------------------------------
#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print() 
print("please waiting")
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")
 

#label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
print("--------------------------------------------------")
print("Before Label Handling ")
print()
print(dataframe_2.head(10))
print("--------------------------------------------------")
print()
print("please waiting")

#3.Data splitting--------------------------------------------------- 

df_train_y=dataframe_2["label"]
df_train_X=dataframe_2.iloc[:,:20]
from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()

df_train_X['proto'] = number.fit_transform(df_train_X['proto'].astype(str))
df_train_X['service'] = number.fit_transform(df_train_X['service'].astype(str))
df_train_X['state'] = number.fit_transform(df_train_X['state'].astype(str))
#df_train_X['attack_cat'] = number.fit_transform(df_train_X['attack_cat'].astype(str))
print("==================================================")
print(" Preprocessing")
print("==================================================")

df_train_X.head(5)
x=df_train_X
y=df_train_y
    

##4.feature selection------------------------------------------------
##kmeans
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y_true = make_blobs(n_samples=175341, centers=4,cluster_std=0.30, random_state=0)
plt.scatter(x[:, 0], x[:, 1], s=20);

kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=20, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.title("k-means")
plt.show()

#---------------------------------------------------------------------------------------
x_train,x_test,y_train,y_test = train_test_split(df_train_X,y,test_size = 0.20,random_state = 42)

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators = 100)  
rf.fit(x_train, y_train)
rf_prediction = rf.predict(x_test)
Result_3=accuracy_score(y_test, rf_prediction)*100
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print("Random Forest")
print()
print(metrics.classification_report(y_test,rf_prediction))
print()
print("Random Forest Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, rf_prediction)
print(cm2)
print("-------------------------------------------------------")
print()
print("please waiting")
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()



#---------------------------------------------------------------------------------------------


from sklearn.tree import DecisionTreeClassifier 
dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
dt.fit(x_train, y_train)
dt_prediction=dt.predict(x_test)
print()
print("---------------------------------------------------------------------")
print("Decision Tree")
print()
print("please waiting")
Result_2=accuracy_score(y_test, dt_prediction)*100
print(metrics.classification_report(y_test,dt_prediction))
print()
print("DT Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, dt_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
print("please waiting")
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm1, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
#ROC graph

#------------------------------------------------------------------------------

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import GradientBoostingClassifier
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
gradient_booster.get_params()

gradient_booster.fit(x_train,y_train)
gb_prediction = gradient_booster.predict(x_test)

print(classification_report(y_test,gradient_booster.predict(x_test)))

Result_2=accuracy_score(y_test, gb_prediction)*100
print()
print("gradient_booster Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, dt_prediction)
print(cm1)
print("-------------------------------------------------------")
print()
print("please waiting")
#------------------------------------------------------------------------------

"Navie Bayies "
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results



y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Navie Bayies Accuracy is:",Result_2,'%')
print()
print("Confusion Matrix:")
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_pred)
print(cm1)
print("-------------------------------------------------------")
print()
print("please waiting")

#---------------------------------------------------------------------------------------------
import easygui
#from easygui import *
Key = "Enter the DOS  Id to be Search"
  
# window title
title = "DOS  Fault Id "
# creating a integer box
str_to_search1=easygui.enterbox(Key, title)
input = int(str_to_search1)


import tkinter as tk
if (rf_prediction[input] ==0 ):
    print("Non Attack ")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "Non Attack ")
    tk.mainloop()
elif (rf_prediction[input] ==1 ):
    print("Attack ")
    root = tk.Tk()
    T = tk.Text(root, height=20, width=30)
    T.pack()
    T.insert(tk.END, "Attack ")
    tk.mainloop()
    #import smtplib as smtp
    import sqlite3
    connection = smtp.SMTP_SSL('smtp.gmail.com', 465)
        
    email_addr = 'sathyakumar17112022@gmail.com'
    email_passwd = 'ncfplwzacztjnxyp'
    connection.login(email_addr, email_passwd)
    connection.sendmail(from_addr=email_addr, to_addrs='sathyakumar17112022@gmail.com', msg="Attack kindly prevent system ")
    connection.close()

