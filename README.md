# CS474 Term Project

## Execution

### 1. Data processing

Data processing processes are divided into two section.

* First Data Processing

  At first, we have to separate dataset into some clusters before issue trend analysis. This time, we separate clusters into 30 pieces based on K-means method. During the issue trend analysis, re-clustering is implemented.

* Second Data Processing

  Second Data Processing is for issue tracking analysis. In Issue Tracking Analysis, we need more specific topic clusters based on first clustering. The algorithms of data processing is as same as the first one.

In short, our procedure can be summarized as follows:

`Data Processing1` -> `Issue Trend Analysis(re clustering)` -> `Data Processing2` -> `Issue Tracking Analysis`

### 2. Issue Trend Analysis

* How to run?

  ```shell
  $> pwd
  home/user/app
  $> python issue_trend_analysis.py
  ```

* Result

  ```text
  [nltk_data] Downloading package stopwords to /root/nltk_data...
  [nltk_data]   Package stopwords is already up-to-date!
  [nltk_data] Downloading package wordnet to /root/nltk_data...
  [nltk_data]   Package wordnet is already up-to-date!
  =======================
           2015
  =======================
  ('Military Parade', 2687) ('Nuclear Weapons', 743) ('Sex Slavery', 657) ('Fishing Boat', 538) ('Arrest Warrant', 477) ('Opposition Party', 451) ('Fourth Largest', 360) ('Sex Crimes', 202) ('Survey Showed', 202) ('Rival Parties', 192) 
  =======================
           2016
  =======================
  ('Corruption Scandal', 709) ('Presidential Race', 404) ('Data Showed', 384) ('Sex Slavery', 303) ('Wartime Sexual', 264) ('Expo 2016', 245) ('Nuclear Test', 238) ('Opposition Parties', 234) ('College Student', 206) ('Found Dead', 204) 
  =======================
           2017
  =======================
  ('President President', 1419) ('Nuclear Test', 1417) ('Corruption Scandal', 645) ('Scientists Develop', 436) ('Top Diplomat', 383) ('Winter Olympics', 368) ('Half Brother', 327) ('Presidential Election', 314) ('Ferry Sinking', 287) ('Conservative Presidents', 252) 
  ```

  Overall format is `(Topic, number of documents)`. The printing format can be revised as you want.

### 3. Issue Tracking Analysis - On Issue

* How to run?

  ```shell
  $> python issue_tracking_one_issue.py 
  ```

* Result

  ```te
  ...
  ...
  [Issue]
  Survey Showed
  
  [On-Issue Event]
  Women Cancer -> Homes Jump -> Elderly Women -> Job Seekers -> Cancer Patients -> Exported 90 -> Taxes 2013 -> Worst Drought -> Rise 12 -> Second Longest -> Salaried Workers -> High Unemployment
  [Detailed Information(per envent)]
  Event: Exported 90(5)
          - Person: Daum Kakao, Sejong, Jeju
          - Organization: Naver, The Coast Guard, Busan
          - Place: Jeju Island
  
  Event: Women Cancer(6)
          - Person: Claire Lee, Yoon Min - Sik, Song Yoon - Mi
          - Organization: Oecd, The Education Ministry, Ikea
          - Place: 
  
  ...
  ...
  ```

  

### 4. Issue Tracking Analysis - Related Issue

* How to run?

  ```shell
  $> python issue_tracking_related_issue.py
  ```

* Result

  ```text
  ```

  

### Data

* `data/koreaherald_201x.csv`: results of first clustering
* `data/recluster/*`: results of re-clustering in issue trend analysis
* `data/sub_cluster/201x/*`: result of second clustering

