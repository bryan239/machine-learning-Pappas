#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import statistics
import scipy.stats
from sklearn.model_selection import train_test_split


# In[2]:


# φόρτωση δεδομένων
ded=pd.read_excel('CogTEL_new.xlsx')


# In[3]:


# απαιτούμε την εμφάνιση ολόκληρου του συνόλου δεδομένων 
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


# In[4]:


# μετατροπή αλφαριθμητικών σε float και παράλληλα μετατροπή "-" σε NaN
ded['education']=pd.to_numeric(ded['education'],errors='coerce')
ded['GDS_6']=pd.to_numeric(ded['GDS_6'],errors='coerce')
ded['GDS_8']=pd.to_numeric(ded['GDS_8'],errors='coerce')
ded['GDS_9']=pd.to_numeric(ded['GDS_9'],errors='coerce')
ded['NPIJ32F']=pd.to_numeric(ded['NPIJ32F'],errors='coerce')


# In[5]:


# για να μπορώ να εξάγω τις συγκεκριμένες στήλες (αχρείαστο βήμα...)
ded = ded.rename(columns={'Antidepressants ': 'Antidepressants' })
ded = ded.rename(columns={'Concentration/attention' : 'Concentration_attention'})


# # Missing values

# In[6]:


rows_with_nan = [index for index, row in ded.iterrows() if row.isnull().any()]

print(rows_with_nan)


# In[7]:


# εμφάνιση missing values ανά μεταβλητή
ded.isna().sum()


# In[8]:


# αφαίρεση στιγμιοτύπων με missing values
ded=ded.dropna()


# # Boxplots

# In[9]:


sns.boxplot(y=ded['education'])


# In[10]:


sns.boxplot(x=ded['Sex'],y=ded['Age'])


# In[11]:


sns.boxplot(y=ded['Age'])


# In[12]:


ded['Age'].mean()


# In[13]:


# αφαίρεση των παραδειγμάτων ηλικίας 50 και κάτω
dedomena=ded[ded['Age']>49]


# In[14]:


sns.boxplot(y=dedomena['Age'])


# In[15]:


dedomena['Age'].mean()


# In[16]:


ded.shape


# In[17]:


# το πλήθος παρατηρήσεων των δεδομένων με την αφαίρεση των outliers
dedomena.shape


# # Περιγραφικά στατιστικά 

# In[18]:


# To έγκυρο σύνολο δεδομένων με την αφαίρεση των outliers και missing values ονομάζεται dedomena
dedomena.info()


# In[19]:


dedomena.describe()


# In[20]:


dedomena.isna().sum()


# In[21]:


a=dedomena[["Sex", "Age"]].groupby("Sex").mean()
b=dedomena[["Sex", "Age"]].groupby("Sex").std()
a,b


# In[22]:


c=dedomena[["Sex", "education"]].groupby("Sex").mean()
d=dedomena[["Sex", "education"]].groupby("Sex").std()
c,d


# In[23]:


dedomena.groupby("Sex")["Sex"].count()


# In[24]:


dedomena.groupby("Age")["Age"].count()


# In[25]:


# μέση ηλικία ανα διάγνωση (εναλλακτικά μπορεί να υπολογιστεί πιο σύνθετα όπως παρουσιάζεται πιο κάτω)
dedomena.groupby("diagnosis")["Age"].mean()


# In[26]:


dedomena.groupby("diagnosis")["Age"].std()


# In[27]:


# μέσος χρόνος εκπαίδευσης ανα κλάση διάγνωσης
dedomena.groupby("diagnosis")["education"].mean()


# In[28]:


dedomena.groupby("diagnosis")["education"].std()


# # Συσχετίσεις

# In[29]:


# correlation matrix
corrMatrix = dedomena.corr()
fig, ax = plt.subplots(figsize=(40,40))
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[30]:


data=dedomena[['Age','education']]
sns.pairplot(data)


# In[31]:


data=dedomena[['Age','diagnosis']]
sns.pairplot(data)


# In[32]:


data=dedomena[['Age','diagnosis','Sex']]
sns.pairplot(data,hue='diagnosis')


# In[33]:


sns.distplot(dedomena['Age'],kde=False)


# In[34]:


gender = dedomena['Sex'].value_counts()
plt.figure(figsize=(7, 6))
ax = gender.plot(kind='bar', rot=0, color="c")

ax.set_xlabel('Sex')
ax.set_ylabel('Number of People')
ax.set_xticklabels(('Male', 'Female'))

for rect in ax.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')    
plt.show()


# In[35]:


sns.distplot(dedomena['education'],kde=False,bins=40)


# In[36]:


imbalance = dedomena['diagnosis'].value_counts()
plt.figure(figsize=(5, 5))
ax = imbalance.plot(kind='bar', rot=0, color="c")

ax.set_xlabel('classes of diagnosis')
ax.set_ylabel('Number of People')


for rect in ax.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 2
    space = 1
    label = format(y_value)
    ax.annotate(label, (x_value, y_value), xytext=(0, space), textcoords="offset points", ha='center', va='bottom')    
plt.show()


# # Έλεγχοι υποθέσεων:
# 

# In[37]:


# ΕΚ ΤΩΝ ΠΡΟΤΕΡΩΝ ΑΝΑΛΥΣΗ ΓΙΑ ΤΟ ΑΝ ΥΠΑΡΧΟΥΝ ΔΙΑΦΟΡΕΣ ΜΕΤΑΞΥ ΤΩΝ ΟΜΑΔΩΝ
# Έλεγχος υπόθεσης: Αge vs Sex
# Xωρίζουμε ανα φύλο τους ασθενείς
groups1=dedomena.groupby("Sex").groups
groups1


# In[38]:


# δημιουργούμε δύο μεταβλητές στις οποίες εκχωρούνται οι ηλικίες των ανδρ΄ών και γυναικών
Age=dedomena['Age']
group_men=Age[groups1[1]]
group_women=Age[groups1[2]]


# In[39]:


# μη παραμετρικός έλεγχος κανονικότητας (Shapiro-Wilk) για τον έλεγχο Age vs Sex
from scipy.stats import shapiro
x=shapiro(group_men)
y=shapiro(group_women)
x,y


# In[40]:


# μη παραμετρικός έλεγχος Αge vs Sex (Mann-Whitney U test ή Wilcoxon rank sum test, που είναι το μη παραμετρικό αναάλογο του t-test)
import scipy.stats as stats
a_elegxos=stats.mannwhitneyu(x=dedomena['Sex'], y=dedomena['Age'], alternative = 'two-sided')
a_elegxos


# In[41]:


# Έλεγχος υπόθεσης: education vs Sex
# δημιουργούμε δύο μεταβλητές στις οποίες εκχωρούνται τα έτη εκπαίδευσης των ανδρ΄ών και γυναικών
dedomena['education']=dedomena['education'].astype(int)
education=dedomena['education']
group_men1=education[groups1[1]]
group_women2=education[groups1[2]]


# In[42]:


# μη παραμετρικός έλεγχος κανονικότητας (Shapiro-Wilk) για τον έλεγχο education vs Sex
z=shapiro(group_men1)
w=shapiro(group_women2)
z,w


# In[43]:


# μη παραμετρικός έλεγχος education vs Sex
b_elegxos=stats.mannwhitneyu(x=dedomena['Sex'], y=dedomena['education'], alternative = 'two-sided')
b_elegxos


# In[44]:


# Έλεγχος υπόθεσης: diagnosis vs Age
# Xωρίζουμε ανα ομάδα διάγνωσης τους ασθενείς
groups2=dedomena.groupby("diagnosis").groups
groups2


# In[45]:


# Ανάθεση των αποτελεσμάτων της ομαδοποίησης που έγινε παραπάνω, σε σχέση με την μεταβλητή Age.
# Δηλαδή έχουμε για κάθε group_i, τις ηλικίες των ασθενών που ανήκουν στην κλάση διάγνωσης i, i=0, 1, 2, 3. 
Age=dedomena['Age']
group_0=Age[groups2[0]]
group_1=Age[groups2[1]]
group_2=Age[groups2[2]]
group_3=Age[groups2[3]]


# In[46]:


# έλεγχος κανονικότητας, προκειμένου να ελέγξουμε αν μπορούμε να κάνουμε One-way Anova
shapiro(group_0)


# In[47]:


shapiro(group_1)


# In[48]:


shapiro(group_2)


# In[49]:


shapiro(group_3)


# In[50]:


# Υπολογισμός μέσης τιμής και τυπικής απόκλισης για την μεταβλητή Age με την κλάση διάγνωσης "0"(ανάλογα αποτελέσματα δίνονται για "1", "2" και "3".)
d1=group_0.mean()
d2=group_0.std()
d1,d2


# In[51]:


# Kruskal-Wallis Test (είναι το μη παραμετρικό ανάλογo του One-way ANOVA)
stats.kruskal(group_0, group_1, group_2, group_3)


# In[52]:


# Post hoc analysis με τον έλεγχο Dunn
get_ipython().system('pip install scikit-posthocs')
import scikit_posthocs as sp


# In[53]:


# ΕΚ ΤΩΝ ΥΣΤΕΡΩΝ ΑΝΑΛΥΣΗ ΓΙΑ ΤΟ ΠΟΥ ΑΚΡΙΒΩΣ ΥΠΑΡΧΟΥΝ ΔΙΑΦΟΡΕΣ
# diagnosis vs Age (post hoc with Dunn)
data=[group_0, group_1, group_2, group_3]
sp.posthoc_dunn(data)


# In[67]:


# X^2 έλεγχος για diagnosis vs Sex
# πίνακας συνάφειας 
table = pd.crosstab(dedomena.Sex, dedomena.diagnosis, margins=True)
table


# In[68]:


# πίνακας συχνοτ΄ήτων
freq_table= table/len(dedomena)
  
freq_table


# In[65]:





# In[55]:


# Με ενδιαφέρει μόνο το p-value του ελέγχου άρα εμφανίζω μόνο αυτό
from scipy.stats import chi2_contingency 
statistic, pvalue, dfreedom, array1=stats.chi2_contingency(table)
pvalue


# In[56]:


# X^2 έλεγχος για diagnosis vs education (θεωρώντας κατηγορική την education)
# πίνακας συνάφειας 
table1 = pd.crosstab(dedomena.diagnosis, dedomena.education, margins=True)
table1


# In[57]:


statistic2, pvalue2, dfreedom2, array2=stats.chi2_contingency(table1)
pvalue2


# In[58]:


# θεωρούμε συνεχή την μεταβλητή education που έχει διακριτοποιηθεί
# Εξετάζουμε αν εφαρμόζεται Anova για diagnosis vs education, μέσω των ελέγχων για Κανονικότητα (Shapiro-Wilk tests)
# έλεγχος κανονικότητας εντός των υποομάδων "0", "1", "2" και "3", σε σχέση με την μεταβλητή education
education=dedomena['education']
group0_0=education[groups2[0]]
group1_1=education[groups2[1]]
group2_2=education[groups2[2]]
group3_3=education[groups2[3]]


# In[59]:


shapiro(group0_0)


# In[60]:


shapiro(group1_1)


# In[61]:


shapiro(group2_2)


# In[62]:


shapiro(group3_3)


# In[63]:


# Kruskal-Wallis Test 
stats.kruskal(group0_0, group1_1, group2_2, group3_3)


# In[64]:


# diagnosis vs education (post-hoc analysis with Dunn)
data1=[group0_0, group1_1, group2_2, group3_3]
sp.posthoc_dunn(data1)


# In[ ]:





# In[ ]:




