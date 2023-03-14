project
---

---

<div class="cell markdown" id="9xZnRXM7x0Cv">

# CUHK-STAT1013: Practical Assignment Part 1: Sharing Your Idea and Data

</div>

<div class="cell markdown" id="9Fy05KAkyJI0">

## Titanic dataset background

**Description**:

Dataset describing the survival status of individual passengers on the
Titanic.

**Github**:
<https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>

**Sample size**: 1,309

**Feature documentation**:

| Feature   | Class      | Shape | Dtype   |
|:----------|:-----------|:------|:--------|
| age       | Tensor     |       | float32 |
| boat      | Tensor     |       | string  |
| body      | Tensor     |       | int32   |
| cabin     | Tensor     |       | string  |
| embarked  | ClassLabel |       | int64   |
| fare      | Tensor     |       | float32 |
| home.dest | Tensor     |       | string  |
| name      | Tensor     |       | string  |
| parch     | Tensor     |       | int32   |
| pclass    | ClassLabel |       | int64   |
| sex       | ClassLabel |       | int64   |
| sibsp     | Tensor     |       | int32   |
| survived  | ClassLabel |       | int64   |
| ticket    | Tensor     |       | string  |

</div>

<div class="cell markdown" id="k85zO7zxys4H">

## Hypothesis

-   Tell us what your idea is and why you have chosen to pursue this
    idea.
    -   we are interested in "*Do passengers with higher levels of
        wealth(high Pclass) have a higher probability of survival on the
        Titanic?*"

-   What two groups you are comparing:
    -   **G1**: survival rate of high Pclass(1) on the Titanic; **G2**:
        survival rate of low Pclass(2 and 3) on the Titanic

-   What you will be measuring (i.e., what your response variable will
    be)
    -   `survived`

-   Is your response variable quantitative rather than categorical?
    -   `survived` is a quantitative variable, with the order `1 > 0`

-   Make a prediction about what kind of difference you expect to see
    between your samples and WHY.
    -   We'd expect that **G1** \> **G2** since [Disproportionate
        Devastation](https://courses.bowdoin.edu/history-2203-fall-2020-kmoyniha/reflection/).

-   Talk about how you will gather your data
    -   From Github link:
        <https://github.com/datasciencedojo/datasets/blob/master/titanic.csv>

    -   Search for relevant information on the Internet or in the school
        library

    -   Find documentaries or interviews about the event

    -   Refer to other authoritative research

-   ## If you had unlimited resources (time, money, staff, etc.) how would you collect your data?

    Use better data sources, such as passenger interviews, crew records,
    etc., to ensure that the data is reasonable and truthful.

-   Exclude data subjectivity and ensure randomness as much as possible

-   Collect revelent data on a larger scale while ensuring the
    authenticity of the data for more accurate analysis

-   Collect similar shipwrecks from the same period to find
    commonalities in the data to support conclusions

</div>

<div class="cell markdown" id="3GOdPWT03PQB">

## Prepare your dataset

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:206}"
id="mUxJb4hxvpHQ" outputId="e6f5c734-14b3-4979-b0f9-312c8278bb7d">

``` python
## load dataset from github

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 5]

sns.set()

df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df.head(5)
```

<div class="output execute_result" execution_count="26">

       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   

                                                    Name     Sex   Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  22.0      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                             Heikkinen, Miss. Laina  female  26.0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                           Allen, Mr. William Henry    male  35.0      0   

       Parch            Ticket     Fare Cabin Embarked  
    0      0         A/5 21171   7.2500   NaN        S  
    1      0          PC 17599  71.2833   C85        C  
    2      0  STON/O2. 3101282   7.9250   NaN        S  
    3      0            113803  53.1000  C123        S  
    4      0            373450   8.0500   NaN        S  

</div>

</div>

<div class="cell markdown" id="55xAIxVa3hpQ">

-   Tell us what groups you want to compare in the dataset
    -   **G1** (survived \| Pclass = 1) vs. **G2** (survived \| Pclass =
        2&3)

</div>

<div class="cell markdown" id="13PdL3ht3902">

-   Print first 5 records of each group, respectively.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="UNL0WXav3hLj" outputId="e3c6cb5c-b06f-44e5-ae65-99da33109e2c">

``` python
## First 5 records of G1 (male)
(df[df['Pclass'] == 1 ]['Survived']).head(5)
```

<div class="output execute_result" execution_count="11">

    1     1
    3     1
    6     0
    11    1
    23    1
    Name: Survived, dtype: int64

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="dhe52HVB4T1O" outputId="fc9b953b-cc5d-426f-f3c0-25ad93396bb2">

``` python
## First 5 records of G2 (female)
(df[(df['Pclass'] ==2) | (df['Pclass'] == 3)]["Survived"]).head(5)
```

<div class="output execute_result" execution_count="73">

    0    0
    2    1
    4    0
    5    0
    7    0
    Name: Survived, dtype: int64

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="FrhHy57ATmhV" outputId="eaa4ef5f-2a2c-4f4a-854d-3f0c17d2c4cc">

``` python
print('conditional median of ALL | Pclass')
print(df.groupby('Pclass').median())
```

<div class="output stream stdout">

    conditional median of ALL | Pclass
            PassengerId  Survived   Age  SibSp  Parch     Fare
    Pclass                                                    
    1             472.0       1.0  37.0    0.0    0.0  60.2875
    2             435.5       0.0  29.0    0.0    0.0  14.2500
    3             432.0       0.0  24.0    0.0    0.0   8.0500

</div>

</div>

<div class="cell markdown" id="Fr5_a-DeT2we">

With the groupby analysis, It is easy to see that even though Pclass 1
passengers are older, they are still the only ones in the three Pclasses
with a median above 1, indicating a higher survival rate

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="faS2YFq7F8U_" outputId="0bb3d400-1ab2-4c3b-cfba-50c61469d1f1">

``` python
print(df[df['Pclass'] == 1]['Survived'].mean())
print(df[df['Pclass'] == 2]['Survived'].mean())
print(df[df['Pclass'] == 3]["Survived"].mean())
```

<div class="output stream stdout">

    0.6296296296296297
    0.47282608695652173
    0.24236252545824846

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Mz9vy5YFQiAy" outputId="c5f33352-8f3a-4508-e6f9-4834f30b23e3">

``` python
print('conditional prob of survived | Pclass 1')
print(len(df[(df['Survived'] == 1) & (df['Pclass'] == 1 )])/len(df[df['Survived'] == 1]))

print('conditional prob of survived | Pclass 2')
print(len(df[(df['Survived'] == 1) & (df['Pclass'] == 2 )])/len(df[df['Survived'] == 1]))

print('conditional prob of survived | Pclass 3')
print(len(df[(df['Survived'] == 1) & (df['Pclass'] == 3 )])/len(df[df['Survived'] == 1]))

```

<div class="output stream stdout">

    conditional prob of survived | Pclass 1
    0.39766081871345027
    conditional prob of survived | Pclass 2
    0.2543859649122807
    conditional prob of survived | Pclass 3
    0.347953216374269

</div>

</div>

<div class="cell markdown" id="HrILF___Y1Yq">

The above figures show that Pclass 1 has a much higher survival rate
than other Pclasses and has a higher conditional Probablity to survive.

</div>

<div class="cell markdown" id="CFzs0PRPUuFQ">

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:221}"
id="zEgfWXaKGvNC" outputId="4e71bc6b-d5aa-4638-9cd0-9dde9a39097a">

``` python
## Any other data description and visualization you want to add.
g = sns.FacetGrid(df, col="Pclass")
g.map(sns.histplot, "Survived")
plt.show()
## Open question, be flexible and no example can be provided.
```
![](https://i.328888.xyz/2023/03/14/9QzDA.png)
    
<div class="output display_data">

![](6163453e23094dee733a2bb47a34e61987d37672.png)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:388}"
id="aJyCRJwqAKgf" outputId="e770612c-9cee-4ac1-e6d3-64d4190b7576">
    


``` python
fig=plt.figure()
fig.set(alpha=0.2)

Survived_0=df.Pclass[df.Survived==0].value_counts()
Survived_1=df.Pclass[df.Survived==1].value_counts()
df1=pd.DataFrame({'No survived':Survived_0,'Survived':Survived_1})
df1.plot(kind="bar",stacked=True)
plt.title("survived by Pclass")
plt.xlabel('Pclass')
plt.ylabel('Counts')
plt.show
```
![avatar](https://i.328888.xyz/2023/03/14/9QEJc.png)

</div>

<div class="output display_data">

![](9315d5b76e32363bcb8e9e6201d95e278bf97815.png)

</div>

</div>

<div class="cell markdown" id="6VP2AJ_6a5LW">

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:368}"
id="FD8ramtIE_93" outputId="2990a0cb-1747-4de9-e456-a77109c0290e">

``` python
g = sns.PairGrid(df[[ 'Pclass',"Survived"]])
g.map_upper(sns.scatterplot, s=20)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3)
plt.show()
```

<div class="output display_data">

![avatar](https://i.328888.xyz/2023/03/14/9Q6ro.png)
</div>

</div>

<div class="cell markdown" id="dHETaXtHbaUd">

In the above chart, we can clearly see that passengers with Pclass = 1
are more likely to be rescued, so it is reasonable to assume that
"Pclass" must be one of the factors that affects the final rescue
outcome

</div>
