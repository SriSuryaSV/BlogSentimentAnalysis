# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 00:44:11 2021

@author: Surya
"""

#(STAT 5870) Final Project
#Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import re #Regular expression Package
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from textblob import Word


# Read the dataset
blog=pd.read_csv('blogtext.csv', nrows=50000)
#Basic analysis of the data
blog.isna().any() #There are no null values
blog.info() #Summary of info
blog.drop_duplicates(subset="id") #Removing rows with duplicate id values (if any)
blog.drop_duplicates(subset="text")#Removing rows with duplicate text values (if any)
blog.shape # There are no duplicate data points

#Checking the dataset
##first few rows
blog.head()
##column names
blog.columns
##Data types
blog.dtypes
##Unique values in column
blog.gender.unique()
blog.gender.value_counts() # The subset choosen has 211 females and 189 males.
blog.age.unique()
blog.age.value_counts() 
blog.topic.unique()
blog.topic.value_counts()
blog.sign.unique() # indUnk is the most popular topic
blog.sign.value_counts() # Aquarius  and Scorpio  are the top zodiac-signs of the bloggers
##Checking random bits of data for better understanding
blog.head(n=1)
blog["text"][111]
blog["text"].tail()


#Cleaning of the text

# Dropping "ID" and "Date" columns as they are not useful for my analysis
blog.drop(['id','date'], axis=1, inplace=True)

##Remove all numbers, symbols and extra spaces, keeping only upper case and lower case alphabets
blog['clean_text']=blog['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ',x))
##Turn all text into lower case
blog['clean_text']=blog['clean_text'].apply(lambda x: x.lower())
#Removing all spaces at the beginning and at the end of the text
blog['clean_text']=blog['clean_text'].apply(lambda x: x.strip())

##Word Elimination and deep cleaning of our text
#Removing stop words
stopwords=set(stopwords.words('english'))
blog['clean_text']=blog['clean_text'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords]))
#Remove non-english words and chat shorcut words that will not contribute to our analysis
#We will have only clean english words with correct spellings.
words = set(nltk.corpus.words.words())
blog['clean_text']=blog['clean_text'].apply(lambda x:' '.join(w for w in nltk.wordpunct_tokenize(x) \
         if w.lower() in words or not w.isalpha()))

##Testing the cleaned text
#testing the non-english language filtering
blog["text"][10]
blog['clean_text'][10]
#Verfying the outcome of above text cleaning process
blog["text"][222]
blog['clean_text'][222]
blog["text"][388]
blog['clean_text'][388]

#Categorizing the data for in-depth age-group wise analysis

#A function that assigns age-groups based on age column
def set_age_group(df):
    df["age_group"] = pd.cut(x=df['age'], bins=[10,20,30,40], labels=["Teens","Young-Adult","Middle-Aged"])
    return df
# Call function
blog = set_age_group(blog)
blog[['age','age_group']]

#Trying to understand age distribution
mean_age=blog['age'].mean() 
print('The mean age of the bloggers in my chosen subset is',mean_age)

mean_age=blog['age'].groupby(blog['age_group']).mean() 
print("The average age of bloggers in each age group is",mean_age)
#Visualing age distrubtion of bloggers in my Dataset

sums = blog['age_group'].value_counts()
sums 
axis('equal');
pie(sums, labels=sums.index);
plt.title("Distribution of bloggers across age groups")
show()



#Text Analysis of the entire data

#Polarity
##Defining a function to find polarity
def pol(clean_text):
    return TextBlob(clean_text).sentiment.polarity
## Calculating polarity for the entire data
blog['Polarity'] = blog["clean_text"].apply(pol)
##Visualizing
##Distribution Plot
sns.distplot(blog['Polarity']).set(title='Distribution of Polarity')
sns.distplot(blog['Polarity'][blog["age_group"] == 'Teens']).set(title='Distribution of Polarity in Teens')
sns.distplot(blog['Polarity'][blog["age_group"] == 'Young-Adult']).set(title='Distribution of Polarity in Young Adults')
sns.distplot(blog['Polarity'][blog["age_group"] == 'Middle-Aged']).set(title='Distribution of Polarity in Middle Aged People')
#Strip plot
sns.stripplot(x='age_group', y="Polarity", 
                   hue='gender', data=blog)
#Box plot
sns.boxplot(x='age_group', y="Polarity", 
                 hue='gender', data=blog, palette="Set1")

#Mean value of polarity across age groups
blog['Polarity'].groupby(blog['age_group']).mean() 



blog.Polarity.max()
pos_idx = blog[(blog.Polarity == 1)].index[0]
neg_idx = blog[(blog.Polarity == -1)].index[0]

print('Most Negative Blog:', blog.iloc[neg_idx][['clean_text']][0])
print('Most Positive Blog:', blog.iloc[pos_idx][['clean_text']][0])

#Most Negative Blog Texts
mn=blog['Polarity'].min()
mn
most_negative = blog[blog["Polarity"] == mn]
most_negative["clean_text"].size
# Visualizing frequently used negative words
all_words = " ".join(most_negative.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#Most Positive Blog Texts
mp=blog['Polarity'].max()
most_positive = blog[blog["Polarity"] == mp]
most_positive["clean_text"].size
# Visualizing frequently used negative words
all_words = " ".join(most_positive.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


#Subjectivity
##Defining a function to find Subjectivity
def senti(clean_text):
    return TextBlob(clean_text).sentiment.subjectivity
## Calculating Subjectivity of the entire data
blog['Subjectivity'] = blog["clean_text"].apply(senti)

##Visualizing
sns.distplot(blog['Subjectivity']).set(title='Distribution of Subjectivity')
sns.barplot(x="age_group",y="Subjectivity",data=blog).set(title='Subjectivity across age groups')


##Visualizing
sns.distplot(blog['Subjectivity']).set(title='Distribution of Subjectivity')
sns.distplot(blog['Subjectivity'][blog["age_group"] == 'Teens']).set(title='Distribution of Subjectivity in Teens')
sns.distplot(blog['Subjectivity'][blog["age_group"] == 'Young-Adult']).set(title='Distribution of Subjectivity in Young Adults')
sns.distplot(blog['Subjectivity'][blog["age_group"] == 'Middle-Aged']).set(title='Distribution of Subjectiviity in Middle Aged People')

sns.stripplot(x='age_group', y="Subjectivity", 
                   hue='gender', data=blog)

sns.boxplot(x='age_group', y="Subjectivity", 
                 hue='gender', data=blog, palette="Set1")



blog.Subjectivity.min()
pos_idx
pos_idx_s = blog[(blog.Subjectivity == 1)].index[0]
print('Most Subjective Blog:', blog.iloc[pos_idx_s][['clean_text']][0])

neg_idx_s = blog[(blog.Subjectivity == 0.0)].index[0]
print('Most Objective Blog:', blog.iloc[neg_idx_s][['clean_text']][0])


#Least Subjective Blog Texts
mn_s=blog['Subjectivity'].min()
most_Objectivity = blog[blog["Subjectivity"] == mn_s]
most_Objectivity["clean_text"].size
# Visualizing frequently used negative words
all_words_mn_s = " ".join(most_Objectivity.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_mn_s)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("least subjective")
plt.show()

#Most Subjective Blog Texts
mx_s=blog['Subjectivity'].max()
most_Subjective = blog[blog["Subjectivity"] == mx_s]
most_Subjective["clean_text"].size
# Visualizing frequently used negative words
all_words_mx_s = " ".join(most_Subjective.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_mx_s)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("most subjective")
plt.show()

blog['Subjectivity'].groupby(blog['age_group']).mean() 

# Word Clouds
blog.topic.value_counts()

#Topic "Student" Word Cloud
t_student = blog[blog["topic"] == "Student"]
# Visualizing frequently used words in blog on Student topics
all_words_t_student = " ".join(t_student.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_student)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Student topic")
plt.show()

#Topic "Religion" Word Cloud
t_religion = blog[blog["topic"] == "Religion"]
# Visualizing frequently used words in blog on Student topics
all_words_t_religion = " ".join(t_religion.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_religion)
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Religion topic")
plt.show()

#Age group Word Cloud without removing any words
blog.age_group.value_counts()
#'Young-Adult' Word Cloud
ag='Young-Adult'
t_ag = blog[blog["age_group"] == ag]
# Visualizing frequently used words in blog on Student topics
all_words_t_ag = " ".join(t_ag.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_ag)
plt.imshow(wordcloud)
plt.axis('off')
plt.title(ag)
plt.show()


#'Teens' Word Cloud
ag='Teens'
t_ag = blog[blog["age_group"] == ag]
# Visualizing frequently used words in blog on Student topics
all_words_t_ag = " ".join(t_ag.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_ag)
plt.imshow(wordcloud)
plt.axis('off')
plt.title(ag)
plt.show()

#'Middle-Aged' Word Cloud
ag='Middle-Aged'
t_ag = blog[blog["age_group"] == ag]
# Visualizing frequently used words in blog on Student topics
all_words_t_ag = " ".join(t_ag.clean_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_ag)
plt.imshow(wordcloud)
plt.axis('off')
plt.title(ag)
plt.show()

#Age group Word Cloud by removing certain frequent common words:
blog.age_group.value_counts()
#'Young-Adult' Word Cloud
ag='Young-Adult'
t_ag = blog[blog["age_group"] == ag]
# Visualizing frequently used words in blog on Student topics
all_words_t_ag = " ".join(t_ag.clean_text)
all_words_t_ag= all_words_t_ag.replace('one','')
all_words_t_ag= all_words_t_ag.replace('people','')
all_words_t_ag= all_words_t_ag.replace('time','')
all_words_t_ag= all_words_t_ag.replace('think','')
all_words_t_ag= all_words_t_ag.replace('know','')
all_words_t_ag= all_words_t_ag.replace('well','')
all_words_t_ag= all_words_t_ag.replace('see','')
all_words_t_ag= all_words_t_ag.replace('u','')
all_words_t_ag= all_words_t_ag.replace('going','')
all_words_t_ag= all_words_t_ag.replace('got','')
all_words_t_ag= all_words_t_ag.replace('wold','')
all_words_t_ag= all_words_t_ag.replace('day','')
all_words_t_ag= all_words_t_ag.replace('go','')
all_words_t_ag= all_words_t_ag.replace('s','')
all_words_t_ag= all_words_t_ag.replace('till','')
all_words_t_ag= all_words_t_ag.replace('od','')
all_words_t_ag= all_words_t_ag.replace('really','')
all_words_t_ag= all_words_t_ag.replace('work','')
all_words_t_ag= all_words_t_ag.replace('love','')
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_ag)
plt.imshow(wordcloud)
plt.axis('off')
plt.title(ag)
plt.show()


#'Teens' Word Cloud
ag='Teens'
t_ag = blog[blog["age_group"] == ag]
# Visualizing frequently used words in blog on Student topics
all_words_t_ag = " ".join(t_ag.clean_text)
all_words_t_ag= all_words_t_ag.replace('one','')
all_words_t_ag= all_words_t_ag.replace('people','')
all_words_t_ag= all_words_t_ag.replace('time','')
all_words_t_ag= all_words_t_ag.replace('think','')
all_words_t_ag= all_words_t_ag.replace('know','')
all_words_t_ag= all_words_t_ag.replace('well','')
all_words_t_ag= all_words_t_ag.replace('see','')
all_words_t_ag= all_words_t_ag.replace('u','')
all_words_t_ag= all_words_t_ag.replace('going','')
all_words_t_ag= all_words_t_ag.replace('got','')
all_words_t_ag= all_words_t_ag.replace('wold','')
all_words_t_ag= all_words_t_ag.replace('day','')
all_words_t_ag= all_words_t_ag.replace('go','')
all_words_t_ag= all_words_t_ag.replace('s','')
all_words_t_ag= all_words_t_ag.replace('till','')
all_words_t_ag= all_words_t_ag.replace('od','')
all_words_t_ag= all_words_t_ag.replace('really','')
all_words_t_ag= all_words_t_ag.replace('work','')
all_words_t_ag= all_words_t_ag.replace('love','')
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_ag)
plt.imshow(wordcloud)
plt.axis('off')
plt.title(ag)
plt.show()

#'Middle-Aged' Word Cloud
ag='Middle-Aged'
t_ag = blog[blog["age_group"] == ag]
# Visualizing frequently used words in blog on Student topics
all_words_t_ag = " ".join(t_ag.clean_text)
all_words_t_ag = " ".join(t_ag.clean_text)
all_words_t_ag= all_words_t_ag.replace('one','')
all_words_t_ag= all_words_t_ag.replace('people','')
all_words_t_ag= all_words_t_ag.replace('time','')
all_words_t_ag= all_words_t_ag.replace('think','')
all_words_t_ag= all_words_t_ag.replace('know','')
all_words_t_ag= all_words_t_ag.replace('well','')
all_words_t_ag= all_words_t_ag.replace('see','')
all_words_t_ag= all_words_t_ag.replace('u','')
all_words_t_ag= all_words_t_ag.replace('going','')
all_words_t_ag= all_words_t_ag.replace('got','')
all_words_t_ag= all_words_t_ag.replace('wold','')
all_words_t_ag= all_words_t_ag.replace('day','')
all_words_t_ag= all_words_t_ag.replace('go','')
all_words_t_ag= all_words_t_ag.replace('s','')
all_words_t_ag= all_words_t_ag.replace('till','')
all_words_t_ag= all_words_t_ag.replace('od','')
all_words_t_ag= all_words_t_ag.replace('really','')
all_words_t_ag= all_words_t_ag.replace('work','')
all_words_t_ag= all_words_t_ag.replace('love','')
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white').generate(all_words_t_ag)
plt.imshow(wordcloud)
plt.axis('off')
plt.title(ag)
plt.show()

all_words_t_ag

print(wordcloud.words_.keys())
       
                        
