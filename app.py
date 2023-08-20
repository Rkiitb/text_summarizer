import numpy as np
import pandas as pd
import streamlit as st
import pickle
import nltk
from nltk import word_tokenize , sent_tokenize,PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()
StopWords=stopwords.words('english')
import re
pattern = r'\b\d+\.\d+%|\$\d+\.\d+bn|\w+\b|\$'
def freq_table(text):
    pattern = r'\b\d+\.\d+%|\$\d+\.\d+bn|\w+\b|\$'
    w=re.findall(pattern,text.lower())
    word_freq=dict()
    for x in w:
        x=ps.stem(x)
        if x not in StopWords and len(x)>1:
            if x not in word_freq.keys():
                word_freq[x]=1
            else:
                word_freq[x]+=1
    return word_freq


def frequency_matrix(text):
    freq_matrix = dict()
    for sent in sent_tokenize(text):
        freq_matrix[sent] = freq_table(sent)
    return freq_matrix


def tfmatrix(freq_matrix):
    tf_matrix={}
    for sen,fre_table in freq_matrix.items():
        dic={}
        for w,n in fre_table.items():
            dic[w]=n/len(fre_table)
        tf_matrix[sen]=dic
    return tf_matrix

def Ndct(freq_matrix):
    Nd_word={}
    for sen,fre_table in freq_matrix.items():
        for word in fre_table:
            if word in Nd_word:
                Nd_word[word]+=1
            else:
                Nd_word[word]=1
    return Nd_word


def idf(freq_matrix, ndct):
    idf = {}
    for doc, count in freq_matrix.items():
        d = {}
        for w in count.keys():
            for W, n in ndct.items():
                if w == W:
                    d[w] = np.log10(len(freq_matrix) / (n)) + 1
        idf[doc] = d
    return idf


def tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(), f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

def sent_score(tf_idf_matrix):
    sentscore={}
    for sent,tfidf_mat in tf_idf_matrix.items():
        for value in tfidf_mat.values():
            if sent not in sentscore.keys():
                sentscore[sent]=value
            else:
                sentscore[sent]+=value
    return sentscore

def avg_score(sentscore):
    total_score = sum(sentscore.values())
    return total_score / len(sentscore)


def summary(data):
    txt = ''
    import re
    pattern = r'\b\d+\.\d+%|\$\d+\.\d+bn|\w+\b|\$'
    freqtable = freq_table(data)
    frequency_mat = frequency_matrix(data)
    tf_mat = tfmatrix(frequency_mat)
    ndct = Ndct(frequency_mat)
    idf_mat = idf(frequency_mat, ndct)
    tf_idf_mat = tf_idf_matrix(tf_mat, idf_mat)
    sentscore = sent_score(tf_idf_mat)
    avgscore = avg_score(sentscore)
    for sent, score in sentscore.items():
        if score > avgscore - 0.01:
            txt += sent + ' '
    return txt


st.title('Text Summarization')


text=st.text_area('please provide text')
if len(text)>1000:
    st.write("here's your summary sir/madam")
    st.caption(summary(text))
else:
    st.caption('please provide a long string')
