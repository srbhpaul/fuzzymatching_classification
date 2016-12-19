{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "\"\"\"\n",
    "The concept:\n",
    "\n",
    "In each row of the included datasets, products X and Y are considered to refer to the same security if \n",
    "they have the same ticker, even if the descriptions don't exactly match. The challenge is to use these \n",
    "descriptions to predict whether each pair in the test set also refers to the same security. \n",
    "\"\"\"\n",
    "\n",
    "import re\n",
    "import jellyfish as jf\n",
    "from sklearn import cross_validation\n",
    "import string\n",
    "from sklearn.neighbors import KNeighborsClassifier \n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.grid_search import GridSearchCV\n",
    "from sklearn import svm\n",
    "from sklearn.metrics import f1_score, make_scorer, accuracy_score\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from IPython.display import display\n",
    "\n",
    "train = pd.DataFrame.from_csv('code_challenge_train.csv')\n",
    "test = pd.DataFrame.from_csv('code_challenge_test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>ticker_x</th>\n",
       "      <th>ticker_y</th>\n",
       "      <th>same_security</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>first trust dow jones internet</td>\n",
       "      <td>first trust dj internet idx</td>\n",
       "      <td>FDN</td>\n",
       "      <td>FDN</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>schwab intl large company index etf</td>\n",
       "      <td>schwab strategic tr fundamental intl large co ...</td>\n",
       "      <td>FNDF</td>\n",
       "      <td>FNDF</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>vanguard small cap index adm</td>\n",
       "      <td>vanguard small-cap index fund inst</td>\n",
       "      <td>VSMAX</td>\n",
       "      <td>VSCIX</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>duke energy corp new com new isin #us4 sedol #...</td>\n",
       "      <td>duke energy corp new com new isin #us26441c204...</td>\n",
       "      <td>DUK</td>\n",
       "      <td>DUK</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>visa inc class a</td>\n",
       "      <td>visa inc.</td>\n",
       "      <td>V</td>\n",
       "      <td>V</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                       description_x  \\\n",
       "0                     first trust dow jones internet   \n",
       "1                schwab intl large company index etf   \n",
       "2                       vanguard small cap index adm   \n",
       "3  duke energy corp new com new isin #us4 sedol #...   \n",
       "4                                   visa inc class a   \n",
       "\n",
       "                                       description_y ticker_x ticker_y  \\\n",
       "0                        first trust dj internet idx      FDN      FDN   \n",
       "1  schwab strategic tr fundamental intl large co ...     FNDF     FNDF   \n",
       "2                 vanguard small-cap index fund inst    VSMAX    VSCIX   \n",
       "3  duke energy corp new com new isin #us26441c204...      DUK      DUK   \n",
       "4                                          visa inc.        V        V   \n",
       "\n",
       "  same_security  \n",
       "0          True  \n",
       "1          True  \n",
       "2         False  \n",
       "3          True  \n",
       "4          True  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>same_security</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>semtech corp</td>\n",
       "      <td>semtech corporation</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>vanguard mid cap index</td>\n",
       "      <td>vanguard midcap index - a</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spdr gold trust gold shares</td>\n",
       "      <td>spdr gold trust spdr gold shares</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>vanguard total bond index adm</td>\n",
       "      <td>vanguard total bond market index</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>oakmark international fund class i</td>\n",
       "      <td>oakmark international cl i</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        description_x                     description_y  \\\n",
       "0                        semtech corp               semtech corporation   \n",
       "1              vanguard mid cap index         vanguard midcap index - a   \n",
       "2         spdr gold trust gold shares  spdr gold trust spdr gold shares   \n",
       "3       vanguard total bond index adm  vanguard total bond market index   \n",
       "4  oakmark international fund class i        oakmark international cl i   \n",
       "\n",
       "   same_security  \n",
       "0            NaN  \n",
       "1            NaN  \n",
       "2            NaN  \n",
       "3            NaN  \n",
       "4            NaN  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this section, we analyze the data and come up with a model that would predict whether (X,Y) pairs conform to the same security in the test set."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### I. Creating data sets for cross-validation and testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# divide the original train set into cross-validation and test set\n",
    "train_cv, test_cv = cross_validation.train_test_split(train, \n",
    "                                                      train_size=0.8, \n",
    "                                                      random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>ticker_x</th>\n",
       "      <th>ticker_y</th>\n",
       "      <th>same_security</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1909</th>\n",
       "      <td>growth fund of america cl a</td>\n",
       "      <td>growth fund of america class a - american fund...</td>\n",
       "      <td>AGTHX</td>\n",
       "      <td>AGTHX</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1697</th>\n",
       "      <td>california wtr svc group inc</td>\n",
       "      <td>ca water service grp</td>\n",
       "      <td>CWT</td>\n",
       "      <td>CWT</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1361</th>\n",
       "      <td>microsoft corporation cmn</td>\n",
       "      <td>microsoft corporation</td>\n",
       "      <td>MSFT</td>\n",
       "      <td>MSFT</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2002</th>\n",
       "      <td>vanguard total international bond etf</td>\n",
       "      <td>vanguard total intl bond index etf</td>\n",
       "      <td>BNDX</td>\n",
       "      <td>BNDX</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>910</th>\n",
       "      <td>vang sm cap idx adm</td>\n",
       "      <td>vanguard small-cap index fund inst</td>\n",
       "      <td>VSMAX</td>\n",
       "      <td>VSCIX</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              description_x  \\\n",
       "1909            growth fund of america cl a   \n",
       "1697           california wtr svc group inc   \n",
       "1361              microsoft corporation cmn   \n",
       "2002  vanguard total international bond etf   \n",
       "910                     vang sm cap idx adm   \n",
       "\n",
       "                                          description_y ticker_x ticker_y  \\\n",
       "1909  growth fund of america class a - american fund...    AGTHX    AGTHX   \n",
       "1697                               ca water service grp      CWT      CWT   \n",
       "1361                              microsoft corporation     MSFT     MSFT   \n",
       "2002                 vanguard total intl bond index etf     BNDX     BNDX   \n",
       "910                  vanguard small-cap index fund inst    VSMAX    VSCIX   \n",
       "\n",
       "     same_security  \n",
       "1909          True  \n",
       "1697          True  \n",
       "1361          True  \n",
       "2002          True  \n",
       "910          False  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_cv.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Number of True examples in the train set\n",
    "train_true = float(train_cv.loc[train_cv['same_security']==True].shape[0])\n",
    "# Number of False examples in the train set\n",
    "train_false = float(train_cv.loc[train_cv['same_security']==False].shape[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7478108581436077"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# fraction of true examples\n",
    "train_true/train_cv.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Thus, the data set is not balanced, and has larger number of *True* examples. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def df_transform(df):\n",
    "    \"\"\"\n",
    "    Input: Data Frame\n",
    "    Output: Transformed Data Frame (see below)\n",
    "    \n",
    "    Transformation:\n",
    "    Removes punctuation marks from the descriptions\n",
    "    since, when comparing descriptions either using term freuency, \n",
    "    or other methods, presence or absence of punctuation is not that critical,\n",
    "    and, in fact, can be misleading.\n",
    "    I choose not to lemmatize the words since these are mostly\n",
    "    corporation names, and not normal english words.\n",
    "    \"\"\"\n",
    "    string_x = []\n",
    "    string_y = []\n",
    "    punc = list(string.punctuation)\n",
    "    for row in df.itertuples():\n",
    "        x = [e for e in row[1] if e not in punc]\n",
    "        y = [e for e in row[2] if e not in punc]\n",
    "        string_x.append(\"\".join(x))\n",
    "        string_y.append(\"\".join(y))\n",
    "    \n",
    "    df.loc[:,'x_nopunc'] = string_x\n",
    "    df.loc[:,'y_nopunc'] = string_y\n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# transform train data\n",
    "train_cv = df_transform(train_cv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>ticker_x</th>\n",
       "      <th>ticker_y</th>\n",
       "      <th>same_security</th>\n",
       "      <th>x_nopunc</th>\n",
       "      <th>y_nopunc</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1909</th>\n",
       "      <td>growth fund of america cl a</td>\n",
       "      <td>growth fund of america class a - american fund...</td>\n",
       "      <td>AGTHX</td>\n",
       "      <td>AGTHX</td>\n",
       "      <td>True</td>\n",
       "      <td>growth fund of america cl a</td>\n",
       "      <td>growth fund of america class a  american funds mf</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1697</th>\n",
       "      <td>california wtr svc group inc</td>\n",
       "      <td>ca water service grp</td>\n",
       "      <td>CWT</td>\n",
       "      <td>CWT</td>\n",
       "      <td>True</td>\n",
       "      <td>california wtr svc group inc</td>\n",
       "      <td>ca water service grp</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1361</th>\n",
       "      <td>microsoft corporation cmn</td>\n",
       "      <td>microsoft corporation</td>\n",
       "      <td>MSFT</td>\n",
       "      <td>MSFT</td>\n",
       "      <td>True</td>\n",
       "      <td>microsoft corporation cmn</td>\n",
       "      <td>microsoft corporation</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2002</th>\n",
       "      <td>vanguard total international bond etf</td>\n",
       "      <td>vanguard total intl bond index etf</td>\n",
       "      <td>BNDX</td>\n",
       "      <td>BNDX</td>\n",
       "      <td>True</td>\n",
       "      <td>vanguard total international bond etf</td>\n",
       "      <td>vanguard total intl bond index etf</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>910</th>\n",
       "      <td>vang sm cap idx adm</td>\n",
       "      <td>vanguard small-cap index fund inst</td>\n",
       "      <td>VSMAX</td>\n",
       "      <td>VSCIX</td>\n",
       "      <td>False</td>\n",
       "      <td>vang sm cap idx adm</td>\n",
       "      <td>vanguard smallcap index fund inst</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              description_x  \\\n",
       "1909            growth fund of america cl a   \n",
       "1697           california wtr svc group inc   \n",
       "1361              microsoft corporation cmn   \n",
       "2002  vanguard total international bond etf   \n",
       "910                     vang sm cap idx adm   \n",
       "\n",
       "                                          description_y ticker_x ticker_y  \\\n",
       "1909  growth fund of america class a - american fund...    AGTHX    AGTHX   \n",
       "1697                               ca water service grp      CWT      CWT   \n",
       "1361                              microsoft corporation     MSFT     MSFT   \n",
       "2002                 vanguard total intl bond index etf     BNDX     BNDX   \n",
       "910                  vanguard small-cap index fund inst    VSMAX    VSCIX   \n",
       "\n",
       "     same_security                               x_nopunc  \\\n",
       "1909          True            growth fund of america cl a   \n",
       "1697          True           california wtr svc group inc   \n",
       "1361          True              microsoft corporation cmn   \n",
       "2002          True  vanguard total international bond etf   \n",
       "910          False                    vang sm cap idx adm   \n",
       "\n",
       "                                               y_nopunc  \n",
       "1909  growth fund of america class a  american funds mf  \n",
       "1697                               ca water service grp  \n",
       "1361                              microsoft corporation  \n",
       "2002                 vanguard total intl bond index etf  \n",
       "910                   vanguard smallcap index fund inst  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# this is how the modified train_cv data set looks like\n",
    "train_cv.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### II. Similarity scores between (X, Y) description pairs"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I calculate a similarity score between the adjacent descriptions in X and Y using three measures\n",
    "a) Jaro Winkler \n",
    "b) Levenshtein distance \n",
    "c) cosine similarity based on term frequency. \n",
    "These distances can be computed using the python module called 'jellyfish' and tfidfvectorizer in scikit-learn. I will use these numeric scores later to build a classification model. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def similarity_feature(df):\n",
    "    \"\"\"\n",
    "    Input: Data Frame\n",
    "    Output: Data Frame with additional columns having\n",
    "            Levenshtein distance, Jaro Winkler distance\n",
    "            and cosine similarity based on term frequency\n",
    "    \"\"\"\n",
    "    def jw(x,y):\n",
    "        # Jaro Winkler distance\n",
    "        return jf.jaro_winkler(unicode(x,\"utf-8\"),unicode(y,\"utf-8\"))\n",
    "    def lv(x,y):\n",
    "        # Levenshtein distance\n",
    "        return jf.levenshtein_distance(unicode(x,\"utf-8\"),unicode(y,\"utf-8\"))\n",
    "    def tf(x,y):\n",
    "        # cosine similarity based on term frequency\n",
    "        # use the tf-idf vectorizer in scikit-learn\n",
    "        # I will not use idf, since we are doing string matching\n",
    "        # and not interested whether a word occurs too frequently\n",
    "        # across documents (a measure of its importance)        \n",
    "        tf_transformer = TfidfVectorizer(analyzer=\"word\",use_idf=False)\n",
    "        document = [x,y]\n",
    "        tf_vec = tf_transformer.fit_transform(document)\n",
    "        similarity_score = np.multiply(tf_vec[0],tf_vec[1].T).sum()\n",
    "        return similarity_score\n",
    "        \n",
    "    df.loc[:,'jaro_winkler'] = map(jw,df['x_nopunc'],df['y_nopunc']) \n",
    "    df.loc[:,'levenshtein'] = map(lv,df['x_nopunc'],df['y_nopunc'])   \n",
    "    df.loc[:,'levenshtein'] = df.levenshtein.astype('float64')\n",
    "    df.loc[:,'tf_similarity'] = map(tf,df['x_nopunc'],df['y_nopunc'])\n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_cv = similarity_feature(train_cv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>ticker_x</th>\n",
       "      <th>ticker_y</th>\n",
       "      <th>same_security</th>\n",
       "      <th>x_nopunc</th>\n",
       "      <th>y_nopunc</th>\n",
       "      <th>jaro_winkler</th>\n",
       "      <th>levenshtein</th>\n",
       "      <th>tf_similarity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1909</th>\n",
       "      <td>growth fund of america cl a</td>\n",
       "      <td>growth fund of america class a - american fund...</td>\n",
       "      <td>AGTHX</td>\n",
       "      <td>AGTHX</td>\n",
       "      <td>True</td>\n",
       "      <td>growth fund of america cl a</td>\n",
       "      <td>growth fund of america class a  american funds mf</td>\n",
       "      <td>0.902797</td>\n",
       "      <td>22.0</td>\n",
       "      <td>0.632456</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1697</th>\n",
       "      <td>california wtr svc group inc</td>\n",
       "      <td>ca water service grp</td>\n",
       "      <td>CWT</td>\n",
       "      <td>CWT</td>\n",
       "      <td>True</td>\n",
       "      <td>california wtr svc group inc</td>\n",
       "      <td>ca water service grp</td>\n",
       "      <td>0.792493</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1361</th>\n",
       "      <td>microsoft corporation cmn</td>\n",
       "      <td>microsoft corporation</td>\n",
       "      <td>MSFT</td>\n",
       "      <td>MSFT</td>\n",
       "      <td>True</td>\n",
       "      <td>microsoft corporation cmn</td>\n",
       "      <td>microsoft corporation</td>\n",
       "      <td>0.968000</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.816497</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2002</th>\n",
       "      <td>vanguard total international bond etf</td>\n",
       "      <td>vanguard total intl bond index etf</td>\n",
       "      <td>BNDX</td>\n",
       "      <td>BNDX</td>\n",
       "      <td>True</td>\n",
       "      <td>vanguard total international bond etf</td>\n",
       "      <td>vanguard total intl bond index etf</td>\n",
       "      <td>0.911211</td>\n",
       "      <td>12.0</td>\n",
       "      <td>0.730297</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>910</th>\n",
       "      <td>vang sm cap idx adm</td>\n",
       "      <td>vanguard small-cap index fund inst</td>\n",
       "      <td>VSMAX</td>\n",
       "      <td>VSCIX</td>\n",
       "      <td>False</td>\n",
       "      <td>vang sm cap idx adm</td>\n",
       "      <td>vanguard smallcap index fund inst</td>\n",
       "      <td>0.831898</td>\n",
       "      <td>17.0</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              description_x  \\\n",
       "1909            growth fund of america cl a   \n",
       "1697           california wtr svc group inc   \n",
       "1361              microsoft corporation cmn   \n",
       "2002  vanguard total international bond etf   \n",
       "910                     vang sm cap idx adm   \n",
       "\n",
       "                                          description_y ticker_x ticker_y  \\\n",
       "1909  growth fund of america class a - american fund...    AGTHX    AGTHX   \n",
       "1697                               ca water service grp      CWT      CWT   \n",
       "1361                              microsoft corporation     MSFT     MSFT   \n",
       "2002                 vanguard total intl bond index etf     BNDX     BNDX   \n",
       "910                  vanguard small-cap index fund inst    VSMAX    VSCIX   \n",
       "\n",
       "     same_security                               x_nopunc  \\\n",
       "1909          True            growth fund of america cl a   \n",
       "1697          True           california wtr svc group inc   \n",
       "1361          True              microsoft corporation cmn   \n",
       "2002          True  vanguard total international bond etf   \n",
       "910          False                    vang sm cap idx adm   \n",
       "\n",
       "                                               y_nopunc  jaro_winkler  \\\n",
       "1909  growth fund of america class a  american funds mf      0.902797   \n",
       "1697                               ca water service grp      0.792493   \n",
       "1361                              microsoft corporation      0.968000   \n",
       "2002                 vanguard total intl bond index etf      0.911211   \n",
       "910                   vanguard smallcap index fund inst      0.831898   \n",
       "\n",
       "      levenshtein  tf_similarity  \n",
       "1909         22.0       0.632456  \n",
       "1697         20.0       0.000000  \n",
       "1361          4.0       0.816497  \n",
       "2002         12.0       0.730297  \n",
       "910          17.0       0.000000  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Here is how the training set looks now\n",
    "train_cv.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### III. Building a classification model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "At first, let me modify the held out test set according to the transformation rules used before on the training set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>ticker_x</th>\n",
       "      <th>ticker_y</th>\n",
       "      <th>same_security</th>\n",
       "      <th>x_nopunc</th>\n",
       "      <th>y_nopunc</th>\n",
       "      <th>jaro_winkler</th>\n",
       "      <th>levenshtein</th>\n",
       "      <th>tf_similarity</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1719</th>\n",
       "      <td>cohen &amp; steers real est secs i</td>\n",
       "      <td>cohen &amp; steers real estate secs i</td>\n",
       "      <td>CSDIX</td>\n",
       "      <td>CSDIX</td>\n",
       "      <td>True</td>\n",
       "      <td>cohen  steers real est secs i</td>\n",
       "      <td>cohen  steers real estate secs i</td>\n",
       "      <td>0.974353</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0.800000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>175</th>\n",
       "      <td>chemours company</td>\n",
       "      <td>chemours co</td>\n",
       "      <td>CC</td>\n",
       "      <td>CC</td>\n",
       "      <td>True</td>\n",
       "      <td>chemours company</td>\n",
       "      <td>chemours co</td>\n",
       "      <td>0.937500</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1220</th>\n",
       "      <td>vanguard total bond market index fund admiral ...</td>\n",
       "      <td>vang tot bd mkt inst</td>\n",
       "      <td>VBTLX</td>\n",
       "      <td>VBTIX</td>\n",
       "      <td>False</td>\n",
       "      <td>vanguard total bond market index fund admiral ...</td>\n",
       "      <td>vang tot bd mkt inst</td>\n",
       "      <td>0.619088</td>\n",
       "      <td>33.0</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>562</th>\n",
       "      <td>motley fool independence</td>\n",
       "      <td>motley fool independence fund</td>\n",
       "      <td>FOOLX</td>\n",
       "      <td>FOOLX</td>\n",
       "      <td>True</td>\n",
       "      <td>motley fool independence</td>\n",
       "      <td>motley fool independence fund</td>\n",
       "      <td>0.965517</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.866025</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1184</th>\n",
       "      <td>royal dutch shell plc sponsored adr repstg a shs</td>\n",
       "      <td>royal dutch shell plc</td>\n",
       "      <td>RDS.A</td>\n",
       "      <td>RDS.A</td>\n",
       "      <td>True</td>\n",
       "      <td>royal dutch shell plc sponsored adr repstg a shs</td>\n",
       "      <td>royal dutch shell plc</td>\n",
       "      <td>0.887500</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0.707107</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                          description_x  \\\n",
       "1719                     cohen & steers real est secs i   \n",
       "175                                    chemours company   \n",
       "1220  vanguard total bond market index fund admiral ...   \n",
       "562                            motley fool independence   \n",
       "1184   royal dutch shell plc sponsored adr repstg a shs   \n",
       "\n",
       "                          description_y ticker_x ticker_y same_security  \\\n",
       "1719  cohen & steers real estate secs i    CSDIX    CSDIX          True   \n",
       "175                         chemours co       CC       CC          True   \n",
       "1220               vang tot bd mkt inst    VBTLX    VBTIX         False   \n",
       "562       motley fool independence fund    FOOLX    FOOLX          True   \n",
       "1184              royal dutch shell plc    RDS.A    RDS.A          True   \n",
       "\n",
       "                                               x_nopunc  \\\n",
       "1719                      cohen  steers real est secs i   \n",
       "175                                    chemours company   \n",
       "1220  vanguard total bond market index fund admiral ...   \n",
       "562                            motley fool independence   \n",
       "1184   royal dutch shell plc sponsored adr repstg a shs   \n",
       "\n",
       "                              y_nopunc  jaro_winkler  levenshtein  \\\n",
       "1719  cohen  steers real estate secs i      0.974353          3.0   \n",
       "175                        chemours co      0.937500          5.0   \n",
       "1220              vang tot bd mkt inst      0.619088         33.0   \n",
       "562      motley fool independence fund      0.965517          5.0   \n",
       "1184             royal dutch shell plc      0.887500         27.0   \n",
       "\n",
       "      tf_similarity  \n",
       "1719       0.800000  \n",
       "175        0.500000  \n",
       "1220       0.000000  \n",
       "562        0.866025  \n",
       "1184       0.707107  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# This is the test set with jaro winkler, Levenshtein and cosine similarity scores\n",
    "test_cv = df_transform(test_cv)\n",
    "test_cv = similarity_feature(test_cv)\n",
    "test_cv.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7738927738927739"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# fraction of True examples in the held out test set\n",
    "float(test_cv.loc[test_cv['same_security']==True].shape[0])/test_cv.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Thus, the test_cv set is imbalanced, and a naive prediction can have an accuracy of 77%. The F1-score will be a better metric for judging model accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### A. KNeighbors Classifier with Grid Search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# I use a k-nearest neighbor classifier\n",
    "# with features as the Jaro Winkler, Levenshtein distance and cosine similarity\n",
    "# I do a grid search to find the best parameters\n",
    "# with 5-fold cross-validation\n",
    "features = ['jaro_winkler','levenshtein','tf_similarity']\n",
    "knn = KNeighborsClassifier()\n",
    "parameters = {'n_neighbors':(5,10,15,20,25), 'weights':('distance','uniform')}\n",
    "\n",
    "# We choose f1_score for the False class as a metric\n",
    "# to compare between models\n",
    "f1_false = make_scorer(f1_score,pos_label=0)\n",
    "knn_grid = GridSearchCV(knn,parameters,cv=5,scoring=f1_false)\n",
    "\n",
    "# fit to the train data with five fold cross-validation\n",
    "knn_fit = knn_grid.fit(train_cv.loc[:,features],train_cv.same_security)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'n_neighbors': 5, 'weights': 'distance'}"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Best KNN parameters\n",
    "knn_grid.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.45303867403314918"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# F1-score for False class (used as score function)\n",
    "knn_score = knn_fit.score(test_cv.loc[:,features],test_cv.same_security)\n",
    "knn_predict = knn_fit.predict(test_cv.loc[:,features])\n",
    "knn_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.76923076923076927"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Average prediction accuracy\n",
    "accuracy_score(test_cv.same_security,knn_predict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.85376661742983762"
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# F1-score for True class\n",
    "f1_score(test_cv.same_security,knn_predict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### B. Support Vector Classifier with Grid Search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# I use a Support Vector classifier\n",
    "# with features as the Jaro Winkler, Levenshtein distance and cosine similarity\n",
    "# I do a grid search to find the best parameters\n",
    "# with 5-fold cross-validation\n",
    "features = ['jaro_winkler','levenshtein','tf_similarity']\n",
    "svc = svm.SVC()\n",
    "parameters = {'kernel':('linear','rbf'), 'C':(20,30,35)}\n",
    "\n",
    "# We choose f1_score for the False class as a metric\n",
    "# to compare between models\n",
    "f1_false = make_scorer(f1_score,pos_label=0)\n",
    "svc_grid = GridSearchCV(svc,parameters,cv=5,scoring=f1_false)\n",
    "\n",
    "# fit to the train data with five fold cross-validation\n",
    "svc_fit = svc_grid.fit(train_cv.loc[:,features],train_cv.same_security)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'C': 30, 'kernel': 'rbf'}"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svc_grid.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.35036496350364965"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# F1-score for False class (used as score function)\n",
    "svc_score = svc_fit.score(test_cv.loc[:,features],test_cv.same_security)\n",
    "svc_predict = svc_fit.predict(test_cv.loc[:,features])\n",
    "svc_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.79254079254079257"
      ]
     },
     "execution_count": 99,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Average prediction accuracy\n",
    "accuracy_score(test_cv.same_security,svc_predict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.87656033287101254"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# F1-score for True class\n",
    "f1_score(test_cv.same_security,svc_predict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Result"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Thus, the average prediction accuracy using either of the KNN or SVM classifier is around 77% on the held out test set. This is actually close to the fraction of *True* class population, and hence not a good metric to determine model performance. The F1-score on the *False* class, although less compared to the *True* class, is however, nonzero and is approximately 0.45 for the KNN classifier. We chose the model that maximised this score. We can now use this model to make predictions for the provided test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# transform the test set according to transformations described before\n",
    "test = df_transform(test)\n",
    "test = similarity_feature(test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>same_security</th>\n",
       "      <th>x_nopunc</th>\n",
       "      <th>y_nopunc</th>\n",
       "      <th>jaro_winkler</th>\n",
       "      <th>levenshtein</th>\n",
       "      <th>tf_similarity</th>\n",
       "      <th>predictions</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>semtech corp</td>\n",
       "      <td>semtech corporation</td>\n",
       "      <td>NaN</td>\n",
       "      <td>semtech corp</td>\n",
       "      <td>semtech corporation</td>\n",
       "      <td>0.926316</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>vanguard mid cap index</td>\n",
       "      <td>vanguard midcap index - a</td>\n",
       "      <td>NaN</td>\n",
       "      <td>vanguard mid cap index</td>\n",
       "      <td>vanguard midcap index  a</td>\n",
       "      <td>0.937879</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.577350</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spdr gold trust gold shares</td>\n",
       "      <td>spdr gold trust spdr gold shares</td>\n",
       "      <td>NaN</td>\n",
       "      <td>spdr gold trust gold shares</td>\n",
       "      <td>spdr gold trust spdr gold shares</td>\n",
       "      <td>0.931713</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.956183</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>vanguard total bond index adm</td>\n",
       "      <td>vanguard total bond market index</td>\n",
       "      <td>NaN</td>\n",
       "      <td>vanguard total bond index adm</td>\n",
       "      <td>vanguard total bond market index</td>\n",
       "      <td>0.939532</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>oakmark international fund class i</td>\n",
       "      <td>oakmark international cl i</td>\n",
       "      <td>NaN</td>\n",
       "      <td>oakmark international fund class i</td>\n",
       "      <td>oakmark international cl i</td>\n",
       "      <td>0.945249</td>\n",
       "      <td>8.0</td>\n",
       "      <td>0.577350</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                        description_x                     description_y  \\\n",
       "0                        semtech corp               semtech corporation   \n",
       "1              vanguard mid cap index         vanguard midcap index - a   \n",
       "2         spdr gold trust gold shares  spdr gold trust spdr gold shares   \n",
       "3       vanguard total bond index adm  vanguard total bond market index   \n",
       "4  oakmark international fund class i        oakmark international cl i   \n",
       "\n",
       "   same_security                            x_nopunc  \\\n",
       "0            NaN                        semtech corp   \n",
       "1            NaN              vanguard mid cap index   \n",
       "2            NaN         spdr gold trust gold shares   \n",
       "3            NaN       vanguard total bond index adm   \n",
       "4            NaN  oakmark international fund class i   \n",
       "\n",
       "                           y_nopunc  jaro_winkler  levenshtein  tf_similarity  \\\n",
       "0               semtech corporation      0.926316          7.0       0.500000   \n",
       "1          vanguard midcap index  a      0.937879          4.0       0.577350   \n",
       "2  spdr gold trust spdr gold shares      0.931713          5.0       0.956183   \n",
       "3  vanguard total bond market index      0.939532          9.0       0.800000   \n",
       "4        oakmark international cl i      0.945249          8.0       0.577350   \n",
       "\n",
       "  predictions  \n",
       "0        True  \n",
       "1        True  \n",
       "2        True  \n",
       "3        True  \n",
       "4        True  "
      ]
     },
     "execution_count": 113,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# predict class values using KNN classifier\n",
    "test_predict = knn_fit.predict(test.loc[:,features])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>description_x</th>\n",
       "      <th>description_y</th>\n",
       "      <th>same_security</th>\n",
       "      <th>x_nopunc</th>\n",
       "      <th>y_nopunc</th>\n",
       "      <th>jaro_winkler</th>\n",
       "      <th>levenshtein</th>\n",
       "      <th>tf_similarity</th>\n",
       "      <th>predictions</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>semtech corp</td>\n",
       "      <td>semtech corporation</td>\n",
       "      <td>NaN</td>\n",
       "      <td>semtech corp</td>\n",
       "      <td>semtech corporation</td>\n",
       "      <td>0.926316</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>vanguard mid cap index</td>\n",
       "      <td>vanguard midcap index - a</td>\n",
       "      <td>NaN</td>\n",
       "      <td>vanguard mid cap index</td>\n",
       "      <td>vanguard midcap index  a</td>\n",
       "      <td>0.937879</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.577350</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spdr gold trust gold shares</td>\n",
       "      <td>spdr gold trust spdr gold shares</td>\n",
       "      <td>NaN</td>\n",
       "      <td>spdr gold trust gold shares</td>\n",
       "      <td>spdr gold trust spdr gold shares</td>\n",
       "      <td>0.931713</td>\n",
       "      <td>5.0</td>\n",
       "      <td>0.956183</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>vanguard total bond index adm</td>\n",
       "      <td>vanguard total bond market index</td>\n",
       "      <td>NaN</td>\n",
       "      <td>vanguard total bond index adm</td>\n",
       "      <td>vanguard total bond market index</td>\n",
       "      <td>0.939532</td>\n",
       "      <td>9.0</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>oakmark international fund class i</td>\n",
       "      <td>oakmark international cl i</td>\n",
       "      <td>NaN</td>\n",
       "      <td>oakmark international fund class i</td>\n",
       "      <td>oakmark international cl i</td>\n",
       "      <td>0.945249</td>\n",
       "      <td>8.0</td>\n",
       "      <td>0.577350</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>pfizer inc div: 1.200</td>\n",
       "      <td>pfizer inc com</td>\n",
       "      <td>NaN</td>\n",
       "      <td>pfizer inc div 1200</td>\n",
       "      <td>pfizer inc com</td>\n",
       "      <td>0.872932</td>\n",
       "      <td>8.0</td>\n",
       "      <td>0.577350</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>spartan global ex us index fid adv cl</td>\n",
       "      <td>sptn glb xus idx adv</td>\n",
       "      <td>NaN</td>\n",
       "      <td>spartan global ex us index fid adv cl</td>\n",
       "      <td>sptn glb xus idx adv</td>\n",
       "      <td>0.770811</td>\n",
       "      <td>17.0</td>\n",
       "      <td>0.158114</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>vanguard total bond market idx-adm</td>\n",
       "      <td>vanguard total bond market index fund investor...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>vanguard total bond market idxadm</td>\n",
       "      <td>vanguard total bond market index fund investor...</td>\n",
       "      <td>0.908444</td>\n",
       "      <td>22.0</td>\n",
       "      <td>0.632456</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>banco latinoamericano de exportacio class e co...</td>\n",
       "      <td>banco latinoamericano come-e</td>\n",
       "      <td>NaN</td>\n",
       "      <td>banco latinoamericano de exportacio class e co...</td>\n",
       "      <td>banco latinoamericano comee</td>\n",
       "      <td>0.883367</td>\n",
       "      <td>30.0</td>\n",
       "      <td>0.408248</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>baidu inc fadr 1 adr reps 0.1 ord shs</td>\n",
       "      <td>baidu inc spons ads repr 0.10 ord cls a us0.00005</td>\n",
       "      <td>NaN</td>\n",
       "      <td>baidu inc fadr 1 adr reps 01 ord shs</td>\n",
       "      <td>baidu inc spons ads repr 010 ord cls a us000005</td>\n",
       "      <td>0.839621</td>\n",
       "      <td>21.0</td>\n",
       "      <td>0.353553</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                       description_x  \\\n",
       "0                                       semtech corp   \n",
       "1                             vanguard mid cap index   \n",
       "2                        spdr gold trust gold shares   \n",
       "3                      vanguard total bond index adm   \n",
       "4                 oakmark international fund class i   \n",
       "5                              pfizer inc div: 1.200   \n",
       "6              spartan global ex us index fid adv cl   \n",
       "7                 vanguard total bond market idx-adm   \n",
       "8  banco latinoamericano de exportacio class e co...   \n",
       "9              baidu inc fadr 1 adr reps 0.1 ord shs   \n",
       "\n",
       "                                       description_y  same_security  \\\n",
       "0                                semtech corporation            NaN   \n",
       "1                          vanguard midcap index - a            NaN   \n",
       "2                   spdr gold trust spdr gold shares            NaN   \n",
       "3                   vanguard total bond market index            NaN   \n",
       "4                         oakmark international cl i            NaN   \n",
       "5                                     pfizer inc com            NaN   \n",
       "6                               sptn glb xus idx adv            NaN   \n",
       "7  vanguard total bond market index fund investor...            NaN   \n",
       "8                       banco latinoamericano come-e            NaN   \n",
       "9  baidu inc spons ads repr 0.10 ord cls a us0.00005            NaN   \n",
       "\n",
       "                                            x_nopunc  \\\n",
       "0                                       semtech corp   \n",
       "1                             vanguard mid cap index   \n",
       "2                        spdr gold trust gold shares   \n",
       "3                      vanguard total bond index adm   \n",
       "4                 oakmark international fund class i   \n",
       "5                                pfizer inc div 1200   \n",
       "6              spartan global ex us index fid adv cl   \n",
       "7                  vanguard total bond market idxadm   \n",
       "8  banco latinoamericano de exportacio class e co...   \n",
       "9               baidu inc fadr 1 adr reps 01 ord shs   \n",
       "\n",
       "                                            y_nopunc  jaro_winkler  \\\n",
       "0                                semtech corporation      0.926316   \n",
       "1                           vanguard midcap index  a      0.937879   \n",
       "2                   spdr gold trust spdr gold shares      0.931713   \n",
       "3                   vanguard total bond market index      0.939532   \n",
       "4                         oakmark international cl i      0.945249   \n",
       "5                                     pfizer inc com      0.872932   \n",
       "6                               sptn glb xus idx adv      0.770811   \n",
       "7  vanguard total bond market index fund investor...      0.908444   \n",
       "8                        banco latinoamericano comee      0.883367   \n",
       "9    baidu inc spons ads repr 010 ord cls a us000005      0.839621   \n",
       "\n",
       "   levenshtein  tf_similarity predictions  \n",
       "0          7.0       0.500000        True  \n",
       "1          4.0       0.577350        True  \n",
       "2          5.0       0.956183        True  \n",
       "3          9.0       0.800000        True  \n",
       "4          8.0       0.577350        True  \n",
       "5          8.0       0.577350        True  \n",
       "6         17.0       0.158114       False  \n",
       "7         22.0       0.632456        True  \n",
       "8         30.0       0.408248        True  \n",
       "9         21.0       0.353553        True  "
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.loc[:,'predictions'] = test_predict\n",
    "test.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8062015503875969"
      ]
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# fraction of predicted positives\n",
    "float(test.loc[test['predictions']==True].shape[0])/test.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above fraction is close to the fraction of positive examples in the training set. "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
