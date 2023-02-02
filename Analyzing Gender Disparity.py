# Databricks notebook source
# MAGIC %md
# MAGIC #Analyzing Gender Disparity
# MAGIC Data Scientists use data to understand and explain the phenomena around them. In this case, I will analyze the given data sets in order to gain a greater understanding on how compensation is affected by social factors such as gender, education, and job satisfaction. 
# MAGIC 
# MAGIC This is a multivariate problem that is affected by the interaction of many features and is based on a large international survey. The questions answered during this analysis should shed some light on gender disparity issues, as well as disparities in other areas such as in age and in between countries. In order to compare compensations between countries for this analysis, I will use a file to convert all foreign currencies to the US dollar.
# MAGIC 
# MAGIC The files containing the data being analyzed are:
# MAGIC 
# MAGIC <ul>
# MAGIC   
# MAGIC   <li>survey_responses.csv: Contains about 17,000 answers from all around the world</li>
# MAGIC   
# MAGIC   <li>conversion2dollar.csv: Contains conversion rates to the US dollar </li>
# MAGIC </ul>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Create the DataFrames
# MAGIC 
# MAGIC The first step in analyzing this data is reading the csv files and turning them into dataframes where we can manipulate the data.

# COMMAND ----------

# Here IntegerType is imported which allows for the values in the CompensationAmound field to be converted to Integer type.
from pyspark.sql.types import IntegerType
# Here pyspark.sql.functions is imported which allows for several key functions such as floor, ceiling, concat, lit, etc.
import pyspark.sql.functions as f

# Here the csv files for the survey responses and the conversion rates are uploaded to spark as dataframes with their schema and header intact
responseDF = spark.read.format("csv").option("inferSchema", "true").option("header", "true").option('quote', '"').option('escape', '"').load("dbfs:/FileStore/CUS681/survey_responses.csv")
ratesDF = spark.read.format("csv").option("inferSchema", "true").option("header", "true").load("dbfs:/FileStore/CUS681/conversion2dollar.csv")

# Here all null values are filled with the value 0
responseDF = responseDF.fillna({'CompensationAmount':0})

# Here a new column is created based off the CompensationAmount column 
# This new column removes any commas in any of the values and casts every value in the column as an Integer type
responseDF = responseDF.withColumn("Compensation", f.regexp_replace(f.col("CompensationAmount"), "[,]", "").cast(IntegerType()))

# Here the old column that was used to create the Compensation column is dropped
responseDF = responseDF.drop("CompensationAmount")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Note that the CompensationAmount field is converted into an Integer and is now called "Compensation"
# MAGIC 
# MAGIC I start preprocessing the data by taking a look at a description of some key columns from the survey data file.

# COMMAND ----------

display(responseDF.select("Compensation", "CompensationCurrency", "GenderSelect").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC By taking a look at the description of these rows, I see that the minimum for compensation is -99. This data is incorrect since you cannot have a negative salary. I also see that the count for GenderSelect, CompensationCurrency, and Compensation are not equal to each other, meaning there are some null values for these columns. According to the count for CompensationCurrency there are many null values for that field. However, there is no way to find out what country code those null values should be because the country code will not always be equal to the country for that column. For example, the country can be the United States, but the CompensationCurrency could be from another country such as HKD. Since this is the case when doing analysis on salary, we will only be able to use the rows of data with a country code. However, when doing analysis on columns not dealing with salary, we can include those rows without a country code for the CompensationCurrency field.

# COMMAND ----------

# For all values in the Compensation column that are less than 0, I set them to 0 since there should not be negative values for Compensation
DF2 = responseDF.withColumn("Compensation", f.when(f.col("Compensation")<0 ,0).otherwise(f.col("Compensation")))

# Here I fill all the null values in the GenderSelect column with the value 'Unmentioned'
DF3 = DF2.na.fill(value="Unmentioned",subset=["GenderSelect"])

# Here I fill all the null values in the Compensation column with the value 0
DF4 = DF3.na.fill(value= 0,subset=["Compensation"])

# Here I fill all the null values in the CompensationCurrency column with the value 'Unmentioned'
responseDF2 = DF4.na.fill(value= "Unmentioned",subset=["CompensationCurrency"])

# COMMAND ----------

display(responseDF2.select("Compensation", "CompensationCurrency", "GenderSelect").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC Whereas the previous description table for these three columns had several issues, this description table corrects those issues. For example, the count for all three columns are now equal at 16,715. I accomplished this by setting the null values for each column to a value. The minimum for compensation is no longer a negative number but 0, since I set all negative values to 0 for that field. I set the null values for GenderSelect and CompensationCurrency to Unmentioned. At first I set these null values to other existing values. However, setting these null values into its own value named unmentioned is more accurate since these null values can be any of the existing values and so may skew the data incorrectly if all null values are assigned to a particular existing value.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. What is the Gender Split Among the Participants by Country?
# MAGIC 
# MAGIC The first question I will answer in my analysis of gender disparity is how each gender group is split by country. 
# MAGIC 
# MAGIC The results of this analysis will show how large of a gap there is between gender groups, and whether or not this gap differs between countries.

# COMMAND ----------

# Here I fill all the null values in the country column with the value Unmentioned
responseDF3 = responseDF2.na.fill(value="Unmentioned",subset=["country"])

# This dataframe shows the country and then the number of males in that country from the given data
maleDF = responseDF3.select("country").where(responseDF3.GenderSelect == "Male").groupBy("country").count().withColumnRenamed("count", "Male").orderBy("country")

# This dataframe shows the country and the number of females corresponding to that country
femaleDF = responseDF3.select(f.col("country").alias("country2")).where(responseDF3.GenderSelect == "Female").groupBy("country2").count().withColumnRenamed("count", "Female").orderBy("country2")

# This dataframe shows the country and the number of non-binaries in that country from the given data
nonbinaryDF = responseDF3.select(f.col("country").alias("country3")).where(responseDF3.GenderSelect == "Non-binary, genderqueer, or gender non-conforming").groupBy("country3").count().withColumnRenamed("count", "Nonbinary").orderBy("country3")

# This dataframe shows the country and the number of people with a different identity in that country from the given data
differentDF = responseDF3.select(f.col("country").alias("country4")).where(responseDF3.GenderSelect == "A different identity").groupBy("country4").count().withColumnRenamed("count", "Different").orderBy("country4")

# Here I use an inner join to combine the male and female dataframes based on the country column
mfDF = maleDF.join(femaleDF,maleDF.country == femaleDF.country2, "inner").drop("country2")

# Here I use an outer join to combine my male and female table with the nonbinary table
# I use outer join here to make sure all rows from both tables are included
# I also fill in the null values in the Nonbinary column with 0's
mfnDF = mfDF.join(nonbinaryDF,mfDF.country == nonbinaryDF.country3, "outer").drop("country3").na.fill(value=0)

# Here I use an outer join to combine my male, female, and nonbinary table with the different id table
gendersDF = mfnDF.join(differentDF,mfDF.country == differentDF.country4, "outer").drop("country4").na.fill(value=0)

# COMMAND ----------

# This table shows the genders split among the participants by country
display(gendersDF)

# COMMAND ----------

# Here I create a total column with all the values from each column and each row summed up
gendersDF2 = gendersDF.withColumn("Total", gendersDF.Male+gendersDF.Female+gendersDF.Nonbinary+gendersDF.Different)

# Here I divide the value in each column by the total and multiply it by 100 to get it closer to percent form
gendersDF3 = gendersDF2.withColumn("Males", (gendersDF2.Male/gendersDF2.Total)*100).withColumn("Females", (gendersDF2.Female/gendersDF2.Total)*100).withColumn("Nonbinaries", (gendersDF2.Nonbinary/gendersDF2.Total)*100).withColumn("Diff", (gendersDF2.Different/gendersDF2.Total)*100).drop("Male","Female","Nonbinary","Different","Total")

# Here I round all the numbers to whole numbers
gendersDF4 = gendersDF3.select("*",f.ceil(f.round("Males",0)).alias("Male"),f.ceil(f.round("Females",0)).alias("Female"),f.ceil(f.round("Nonbinaries",0)).alias("Nonbinary"),f.ceil(f.round("Diff",0)).alias("Different")).drop("Males","Females","Nonbinaries","Diff")

# Here I add a percent column so that I can use it to concatenate the % with the other columns
gendersDF5 = gendersDF4.withColumn("percent",f.lit("%"))

# Here I concatenate the values with the percent column to get the values in percent form
percent_gender_splitDF = gendersDF5.select("country",f.concat(gendersDF5.Male,gendersDF5.percent).alias("Male"),f.concat(gendersDF5.Female,gendersDF5.percent).alias("Female"),f.concat(gendersDF5.Nonbinary,gendersDF5.percent).alias("Nonbinary"),f.concat(gendersDF5.Different,gendersDF5.percent).alias("Different"))

# COMMAND ----------

# This table shows the gender split among the participants by country in percent form
display(percent_gender_splitDF)

# COMMAND ----------

# MAGIC %md
# MAGIC From the above table there are some interesting statistics to take note of. The largest gender disparity exists in the Czech Republic with 94% of the participants from the Czech Republic being male, and 6% of the participants being female. The least disparity in Gender belongs to Ireland, Malaysia, and the Philippines, with the split in gender being 68% male and 32% female for Ireland, 68% male, 30% female and 1% different identity for Malaysia, and 68% Male, 26% female, and 6% nonbinary for the Philippines. Besides the Unmentioned countries, the Philippines had the largest percentage of nonbinary people at 6% while Finland and Argentina had the largest percentage of people with a different identity. Overall, the biggest takeaway from this analysis is the Male gender group outnumbers every other gender group combined by a staggering amount. It is important to note that survey involves people with jobs in STEM fields such as Computer Scientists, Engineers, Statisticians etc. This indicates that there are more males than other gender groups in these types of jobs internationally and that there is a global gender disparity.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Join the Data with the Conversion Rates
# MAGIC 
# MAGIC For the purpose of continuing my analysis on gender disparity, I must convert all foreign currencies to the US dollar. This is so that I can compare all salaries from all countries to each other. 
# MAGIC 
# MAGIC I accomplish this conversion of foreign currencies to the US dollar by performing an inner join between the data with the conversion rates and the survey data. 
# MAGIC 
# MAGIC This will create a table with only the rows where there is a specified conversion rate in the CompensationCurrency column, meaning many rows of data gets dropped during this process since there is no way to compare those salaries to the salaries converted to the US dollar.

# COMMAND ----------

# Here I select only the columns from the survey_responses table that is needed in order to simplify it
responseDF4 = responseDF3.select("GenderSelect", "Country", "Age", "CurrentJobTitleSelect", "Compensation", "CompensationCurrency")

# Here I do an inner join of the survey and conversion rate tables
# Since this is an inner join, rows where CompensationCurrency don't match any rows from OriginCountry get dropped
# This is beneficial since it drops a lot of data where there is missing information
# I also drop the id and originCountry columns since they are unnecessary
responseDF5 = responseDF4.join(ratesDF,responseDF4.CompensationCurrency == ratesDF.originCountry, "inner").drop("id","originCountry")

# Here I convert all the salaries into US dollars by multiplying all the salaries by the exchange rate 
responseUSDF = responseDF5.withColumn("CompensationAmountUS",responseDF5["Compensation"]*responseDF5["exchangeRate"]).drop("exchangeRate")

# COMMAND ----------

# MAGIC %md
# MAGIC The below table shows how each compensation/salary is converted to US dollars. Depending on the country code in CompensationCurrency, the value in the Compensation column is multiplied by an exchange rate corresponding to that specific country code. The product of the two numbers is the value in the CompensationAmountUS column which shows the foreign salaries in US dollars.

# COMMAND ----------

# This table shows the salaries in their country's currency and the salary in US dollars
display(responseUSDF.select("CompensationCurrency","Compensation","CompensationAmountUS"))

# COMMAND ----------

# Here I drop unnecessary columns and rename columns to simplify the survey table
responseUSDF2 = responseUSDF.drop("CompensationCurrency","Compensation").withColumnRenamed("CompensationAmountUS", "Salary").withColumnRenamed("GenderSelect", "Gender").withColumnRenamed("CurrentJobTitleSelect", "Job")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Is there a Difference for the Average/Median Salary Among the Different Gender Groups?
# MAGIC 
# MAGIC I progress my analysis of gender disparity by finding the difference in average and median salary between gender groups.
# MAGIC 
# MAGIC The results of this analysis will grant us another perspective into this issue by showing which gender groups have the highest/lowest salary on average as well as each group's median salary.
# MAGIC 
# MAGIC The reason I use the median salary for this analysis is to reduce the effects of outliers which may affect the average.

# COMMAND ----------

# Here I create a temporary view of the simplified surveys table so that I can perform SQL on it
responseUSDF2.createOrReplaceTempView("responseTbl")

# Here I create a temporary view based off a query where I find the average salary for each gender group
spark.sql("CREATE or replace temp view Average as select round(avg(salary)) as Average_Salary,gender as genders from responseTbl group by gender")

# Here I create a temporary view for each gender group ordered by salary and I add an ID column in order to calculate the median
spark.sql("create or replace temp view Male as SELECT ROW_NUMBER() OVER (ORDER BY salary desc) id, gender, salary FROM responseTbl where gender = 'Male'")
spark.sql("create or replace temp view Female as SELECT ROW_NUMBER() OVER (ORDER BY salary desc) id, gender, salary FROM responseTbl where gender = 'Female'")
spark.sql("create or replace temp view Nonbinary as SELECT ROW_NUMBER() OVER (ORDER BY salary desc) id, gender, salary FROM responseTbl where gender = 'Non-binary, genderqueer, or gender non-conforming'")
spark.sql("create or replace temp view Different as SELECT ROW_NUMBER() OVER (ORDER BY salary desc) id, gender, salary FROM responseTbl where gender = 'A different identity'")

# Here I find the number of rows for each table. I will use this number to find the row with the median.
display(spark.sql("select count(*) as Male from responseTbl where gender = 'Male'"))
display(spark.sql("select count(*) as Female from responseTbl where gender = 'Female'"))
display(spark.sql("select count(*) as Nonbinary from responseTbl where gender = 'Non-binary, genderqueer, or gender non-conforming'"))
display(spark.sql("select count(*) as Different from responseTbl where gender = 'A different identity'"))

# COMMAND ----------

# Here I create a function that takes in the number of total rows in a table and outputs the median row
def median(x):
    if(x%2==0):
        return ((x/2) + ((x/2)+1))/2
    else:
        return (x+1)/2

# These numbers I get from performing count aggregation on each table which means the number of rows in the table
print("Male median row location: " + str(median(3860)))
print("Female median row location: " + str(median(610)))
print("Non-binary median row location: " + str(median(25)))
print("Different median row location: " + str(median(27)))

# COMMAND ----------

# Since I get 1930.5 and 305.5 for the median row locations for male and female temporary views respectively, that means I must take the average of the values in the row below and above those numbers in order to find the median 
spark.sql("create or replace temp view Male2 as select gender,round(sum(salary)/2) as Median_Salary from Male where id = 1930 or id = 1931 group by gender")
spark.sql("create or replace temp view Female2 as select gender,round(sum(salary)/2) as Median_Salary from Female where id = 305 or id = 306 group by gender")

# Since I get a whole number for both the Nonbinary and Different temporary views, I simply retrieve the value in the salary column in that specific row to find the median
spark.sql("create or replace temp view Nonbinary2 as select gender,salary as Median_Salary from Nonbinary where id = 13")
spark.sql("create or replace temp view Different2 as select gender,round(salary) as Median_Salary from Different where id = 14")

# Here I create a table called Median and I insert all the median values from each gender group into it in order to have all the median values in one table corresponding to their gender groups
spark.sql("CREATE TABLE Median (gender varchar(255), Median_Salary int)")
spark.sql("Insert into table Median (gender, Median_Salary) select gender, Median_Salary from male2")
spark.sql("Insert into table Median (gender, Median_Salary) select gender, Median_Salary from Female2")
spark.sql("Insert into table Median (gender, Median_Salary) select gender, Median_Salary from Nonbinary2")
spark.sql("Insert into table Median (gender, Median_Salary) select gender, Median_Salary from Different2")

# Here I combine both the average temporary view and the median table using an inner join into one temporary view so that it shows both Average and Median for the corresponding gender groups
spark.sql("create or replace temp view avg_med as select gender as Gender, Average_Salary, Median_Salary from Median inner join Average on Median.gender = Average.genders order by Average_Salary desc")

# Here I use concat to add a $ sign in front of each number and I add a comma in between the numbers as well
spark.sql("create or replace temp view avg_med2 as select gender, if(length(Average_Salary)=8,concat(left(concat('$',ceiling(Average_Salary)),4),',',right(concat('$',ceiling(Average_Salary)),3)),concat(left(concat('$',ceiling(Average_Salary)),3),',',right(concat('$',ceiling(Average_Salary)),3))) as Average_Salary, concat(left(concat('$',ceiling(Median_Salary)),3),',',right(concat('$',ceiling(Median_Salary)),3)) as Median_Salary from avg_med")

# COMMAND ----------

# If I have to run this notebook again and I run into an error concerning the median table and the hive data warehouse I run this line of code in order to remove the median table from the hive data warehouse so that I can create a new median table
#%fs rm -r /user/hive/warehouse/median

# I run this line of code if I need to run this notebook again after I previously ran it in order to drop the median table and create a new one
#spark.sql("drop table median")

# COMMAND ----------

# MAGIC %md
# MAGIC According to the table below, there is a significant difference on the average salary among the different gender groups, with A different identity coming first, Male coming second, Non-binary coming third, and Female coming fourth. Although Male, Non-binary, and Female are relatively close to each other in Average salary, A different identity is much higher than all three. This may be because of outliers because the median salary for each gender group is much closer to each other. A different identity still comes in first but is not as far away from the rest of the gender groups as it was with the average salary. Male still comes in second, but female and non-binary switch, with female coming in third and non-binary coming in fourth. There is not as much of a significant difference in the median salary among the gender groups compared to the average salary. Also, the average salary is higher than the median salary for each gender group. Another interesting thing to note is the least amount of difference in between the average and median salaries is for the female group where the largest difference in between the two salaries is for a different identity. This may indicate that the female group has the least amount of outliers while a different identity has the most.

# COMMAND ----------

# Here I write a query to show the table with the genders, average salary, and median salary and store it in a dataframe
salaryDF = sqlContext.sql("select * from avg_med2")

# Here I display the dataframe
display(salaryDF)

# COMMAND ----------

# MAGIC %md
# MAGIC One alarming takeaway from analyzing the average salary for each gender group is the significant difference between the male and female groups as well as the male and nonbinary groups. Not only do males make up the majority of these jobs in STEM fields globally, but they on average make a significant amount more money than females and non-binaries too, which may deter these genders from going for these types of jobs in the first place. Not only is the average salaries for males higher than the females and nonbinaries, but the median salary as well showing that this isn't due to outliers in the data and further proves how much of a disparity there is in between genders when it comes to these jobs. Not only do males dominate the market for these fields but they also overall make more money than females and non-binaries too. One interesting statistic to take note of however, is that a different identity has a significantly higher average salary than the other gender groups and a higher median salary as well, and this could be due to a lack of data for this gender group, since the data gives only 27 rows containing data on people with a different identity. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Is salary correlated with Age? If so, describe the relationship.
# MAGIC 
# MAGIC Now that my analysis of gender disparity is complete, I now begin to analyze disparities in other areas such as age disparity. 
# MAGIC 
# MAGIC Specifically, I look to see whether salary has a positive, negative, or no correlation with Age.
# MAGIC 
# MAGIC I find this relationship between Age and salary by analyzing both the Age and Salary columns in the data.
# MAGIC 
# MAGIC #### 5.1 Determine what type of relationship (if any) exists Between Age and Salary.

# COMMAND ----------

# Here I create a temporary view of only the Age and Salary columns and I also make sure to order the view by Age
spark.sql("create or replace temp view sal_age as select Age, round(Salary) as Salary from responseTbl order by Age")

# Here I create a query removing null values and showing off all the data in the dataframe and store it in sal_ageDF
sal_ageDF = sqlContext.sql("select * from sal_age where age is not null and salary is not null")

# Here I convert both the age and salary colums into int type in order to calculate the correlation between the two
sal_ageDF2 = sal_ageDF.select(f.col("Age").cast('int').alias("Age"),f.col("Salary").cast('int').alias("Salary"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Average Salary vs. Age
# MAGIC 
# MAGIC According to the graph below, there seems to be a steady increase in salary from age 18 to age 40. From age 40 onwards besides the outlier at age 72, there does not appear to be a clear pattern of increase or decrease in salary. Towards the tail end of the age range there appears to be a decrease in salary as age increases.

# COMMAND ----------

display(sal_ageDF2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Median Salary vs Age
# MAGIC According to the graph below there seems to be a steady increase in salary from age 19 to age 50. From age 50 onwards there appears to mostly be no relationship between age and salary except for an increase from ages 60 to 65 and an outlier for age 72. Besides the outlier for age 72, from age 70 onwards there appears to be a decrease in salary.

# COMMAND ----------

display(sal_ageDF2)

# COMMAND ----------

# MAGIC %md
# MAGIC # Hypothesis from Analysis
# MAGIC From analyzing the graphs of Average Salary vs. Age and Median Salary vs. Age, salary appears to be positively correlated with Age. The reason why I believe they have a positive relationship is because for both graphs it is clearly shown as age increases salary also increases. However for both graphs, once they reach the 40's and 50's this correlation seems to dissipate and from age 70 onwards salary seems to decrease. However, although this is the case, I still hypothesize that age and salary are positively correlated since there is a clear positive correlation for a little more than half of the age range.

# COMMAND ----------

# MAGIC %md
# MAGIC # Results

# COMMAND ----------

# Calculating correlation between columns
sal_ageDF2.stat.corr('Age','Salary')

# COMMAND ----------

# MAGIC %md
# MAGIC According to the calculations, Age and Salary have a correlation of approximately .11. This means that, according to the data, Age and Salary are slightly positively correlated, meaning they have a weak positive relationship with each other. Therefore, as age increases salary is more likely to increase, then it is to decrease. This is one example proving that there is an age disparity for salaries.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 What is the average salary in the US for people in their 20s? And for people in their 50s?
# MAGIC 
# MAGIC In the previous section, I used visualizations for both the average and median salary as well as the correlation to show that age and salary have a positive relationship with each other, in turn proving age disparity for salary. 
# MAGIC 
# MAGIC In this section I will further prove age disparity concering salary by creating a table showing the average salary per age group for people in the US. 
# MAGIC 
# MAGIC I carry out this analysis by transforming each age into their corresponding decade using arithmetic and the <i>floor</i> operation and then grouping each decade together and finding each decade's average salary.

# COMMAND ----------

# Here I create a temporary view of the dataframe sal_ageDF2 in order to perform SQL operations on it
responseUSDF2.createOrReplaceTempView("sal_ageDF")

# Here I select the age and salary of those people only in the United States
spark.sql("create or replace temp view sal_ageDF2 as select Age, Salary, Country from sal_ageDF where Country = 'United States'")

# Here I remove the salary for ages less than 10 to improve the accuracy of the data
spark.sql("create or replace temp view sal_ageDF3 as select * from sal_ageDF2 where Age >= 10")

# Here I first divide the age column by 10. This is so that I can use the floor operation on each age.
# After each number is reduced to their floor, I multiply the age column by 10 to get it's corresponding decade
# I then create a temporary view with this data
spark.sql("create or replace temp view sal_age4 as select 10*floor(age/10) as Age, Salary from sal_ageDF3")

# Since I have transformed each age to their corresponding decade, I now group the ages by decade and find the average for each decade
spark.sql("create or replace temp view sal_age5 as select Age as Decade, round(avg(Salary)) as Salary from sal_age4 group by Age order by Age")

# Here I add some concatenations to each of the columns in the dataframe to make it more presentable
spark.sql("create or replace temp view sal_age6 as select concat(Decade,'s') as Decade, if(length(Salary)=8,concat(left(concat('$',ceiling(Salary)),4),',',right(concat('$',ceiling(Salary)),3)),concat(left(concat('$',ceiling(Salary)),3),',',right(concat('$',ceiling(Salary)),3))) as Salary from sal_age5")

# COMMAND ----------

# MAGIC %md
# MAGIC **According to the table below, the average salary in the US for people in their 20s is $84,867 and the average salary in the US for people in their 50s is $140,880.**
# MAGIC 
# MAGIC From this table, it is clearly shown that salary increases by decade up until the 40s where salary starts to decrease.

# COMMAND ----------

# Displayed is the average salary in the US per decade
display(spark.sql("select * from sal_age6"))

# COMMAND ----------

# MAGIC %md
# MAGIC The table above is solid evidence of age disparity concerning salary internationally. One can clearly see that as the decades increase, the salary also increases until the 40s where it decreases slightly each decade but jumps again in the 70s. Some interesting statistics to take note of concerning this table is salary increases each decade up until the 40s, where it reaches an average salary of $149,628. The maximum average salary is in the 70s with $190,000, although this is most likely due to outliers. The decade with the lowest average salary is the 10s, with a minimum average salary of $50,000. The largest increase in average salary in between decades is from the 10s to the 20s with an increase of $34,867. The 20s to the 30s has an increase in average salary of $33,053. The 30s to the 40s also has a large jump in salary with an increase of $31,708. As a result of this finding, people who work in these STEM fields should expect a significant increase in salary from their 10s to their 20s, 20s to their 30s, and their 30s to their 40s.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. What is the Job with the Highest Compensation?
# MAGIC 
# MAGIC After analyzing disparities in gender and age concerning salaries, I now turn to analyzing disparities in jobs concerning salaries. 
# MAGIC 
# MAGIC Due to the possibility of outliers missrepresenting the given data, I will perform this analysis using the median salary for each job instead of the average salary.
# MAGIC 
# MAGIC Using the median salary will offset the effects of outliers in the data.

# COMMAND ----------

# Since this question asks for the job with the highest median salary, I make a dataframe with only 2 columns: job and salary.
jobDF = responseUSDF.select("CurrentJobTitleSelect","CompensationAmountUS").withColumnRenamed("CurrentJobTitleSelect", "Job").withColumnRenamed("CompensationAmountUS", "Salary").na.drop().orderBy("Job","Salary")

# Here I create a temporary view in order to perform sql operations on the dataframe
jobDF.createOrReplaceTempView("jobDF")

# Here I add an ID column to the dataframe in order to see row number. This will be useful to find the median salary for each job.
# Also, I make sure to group each job together and order by salary to make it possible to locate the median salary/middle row for each job.
spark.sql("create or replace temp view jobDF0 as SELECT ROW_NUMBER() OVER (ORDER BY Job, Salary) id, Job, Salary FROM jobDF")

# Here I create a view that lists each distinct job and the number of rows each job contains
spark.sql("create or replace temp view jobDF2 as select Job, count(*) as Rows from jobDF group by Job order by Job")

# Here I create a column where I calculate the middle row for each job using the number of rows each job has
spark.sql("create or replace temp view jobDF3 as select Job, Rows, if(Rows%2=0,((Rows/2) + ((Rows/2)+1))/2,(Rows+1)/2) as Median_Row from jobDF2")

# Although I've found the middle row for each job, I now need to find the exact location of each middle row in the table with all the jobs
# The first step I take in doing this is finding the row location one spot before a new job starts to appear on the table
# I do this by summing up the previous rows while subtracting the number of rows for the current job
spark.sql("create or replace temp view jobDF4 as select Job, Rows, (sum(Rows) over (order by Job))-Rows as Cumulative_rows, Median_Row from jobDF3")

# Here I find the median row location of each job in the table with all the jobs.
# I do this by summing up the number of rows before the job appears with the median row for that specific job.
spark.sql("create or replace temp view jobDF5 as select Job, cumulative_rows+median_row as Median_Row_Location from jobDF4")

# One problem is that for some of the median row locations I get half a number such as 118.5.
# This means the median for that job is the average salary of row 118 and row 119.
# So my first step in solving this problem is making two columns, one with the floor and one with the ceiling
# This will give me the numbers directly below and above these half numbers which I need to calculate the average in order to find the median
# For the whole numbers this will not make a difference
spark.sql("create or replace temp view jobDF6 as select Job, floor(Median_Row_Location) as floor_Median, ceil(Median_Row_Location) as ceil_Median from jobDF5")

# Here I use an inner join in order to join the table with the location of the median rows and the table with the salaries
# I specifically join the id column from the table with the salaries with the floor and ceiling median locations of the other table
# Since this is an inner join the view will only contain those rows in the floor median and ceil median column
spark.sql("create or replace temp view jobDF7 as select id, jobDF6.Job, floor_Median, ceil_Median, Salary from jobDF6 inner join jobDF0 on jobDF6.floor_Median = jobDF0.id or jobDF6.ceil_Median = jobDF0.id order by id")

# Here I take the average of the floor median salary and ceiling median salary in order to find the median salary for that job.
# For those jobs with the same floor and ceiling median, this will not make a difference.
spark.sql("create or replace temp view jobDF8 as select Job, round(avg(Salary)) as Median_Salary from jobDF7 group by Job order by Median_Salary desc")

# Here I make the table more presentable by adding a $ sign and a comma to the median salary column
spark.sql("create or replace temp view salaryByJobDF as select Job, if(length(median_salary)=8,concat(left(concat('$',ceiling(Median_Salary)),4),',',right(concat('$',ceiling(Median_Salary)),3)),concat(left(concat('$',ceiling(Median_Salary)),3),',',right(concat('$',ceiling(Median_Salary)),3))) as Median_Salary from jobDF8")

# COMMAND ----------

# MAGIC %md
# MAGIC **The job with the highest compensation according to the median salary of each job in the data is the Operations Research Practitioner with a median salary of $106,921.**

# COMMAND ----------

# This table displays the job title and their corresponding median salary
display(spark.sql("select * from salaryByJobDF"))

# COMMAND ----------

# MAGIC %md
# MAGIC Note that each of these job titles are a part of the STEM field. I can conclude from my previous analysis that these jobs have a disproportionate amount of males compared to other gender groups and also that these jobs pay more on average to males than to females or nonbinaries. Thus these jobs have a gender disparity issue.
# MAGIC An interesting statistic to take note of concerning this table is Programmers have the lowest median salary which is $23,753.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. What is the Country with the Highest Compensation?
# MAGIC 
# MAGIC I have analyzed and proven disparities between gender, age, and job concerning salary. For my final analysis I will show disparity between countries concerning salary.
# MAGIC 
# MAGIC The results of this analysis will show which countries have the highest median salary and which countries have the lowest median salary.
# MAGIC 
# MAGIC I use median salary instead of average salary in order to offset the effect that outliers will have.

# COMMAND ----------

# Since this question wants the country with the highest salary, I only keep the relevant columns: country and salary.
countryDF = responseUSDF.select("Country","CompensationAmountUS").withColumnRenamed("CompensationAmountUS", "Salary").na.drop().orderBy("Country","Salary")

# Here I create a view so I can perform sql operations on the data
countryDF.createOrReplaceTempView("countryDF")

# Here I add an id column to the data as well as grouping the data by country and ordering by salary so I can identify the median row for each country
spark.sql("create or replace temp view countryDF0 as SELECT ROW_NUMBER() OVER (ORDER BY Country, Salary) id, Country, Salary FROM countryDF")

# Here I get the number of rows each country has
spark.sql("create or replace temp view countryDF2 as select Country, count(*) as Rows from countryDF group by Country order by Country")

# Here I calculate the middle row of each country using the number of rows each country has
spark.sql("create or replace temp view countryDF3 as select Country, Rows, if(Rows%2=0,((Rows/2) + ((Rows/2)+1))/2,(Rows+1)/2) as Median_Row from countryDF2")

# Here I find the row number before each job occurs in the table with all the jobs by summing up previous rows and subtracting from current rows
spark.sql("create or replace temp view countryDF4 as select Country, Rows, (sum(Rows) over (order by Country))-Rows as Cumulative_rows, Median_Row from CountryDF3")

# Here I find the median row location by adding the row number before each job occurs with the middle row of each job
spark.sql("create or replace temp view countryDF5 as select Country, cumulative_rows+median_row as Median_Row_Location from countryDF4")

# Since some row locations are half numbers, I use floor and ceil to find the whole numbers below and above the half numbers
spark.sql("create or replace temp view countryDF6 as select Country, floor(Median_Row_Location) as floor_Median, ceil(Median_Row_Location) as ceil_Median from countryDF5")

# Here I use an inner join to select only the salaries with their id equal to the floor and ceiling median locations
spark.sql("create or replace temp view countryDF7 as select id, countryDF6.Country, floor_Median, ceil_Median, Salary from countryDF6 inner join countryDF0 on countryDF6.floor_Median = countryDF0.id or countryDF6.ceil_Median = countryDF0.id order by id")

# Here I find the average of the floor and ceiling salaries for each country in order to find the median salary for each country
spark.sql("create or replace temp view countryDF8 as select country, round(avg(Salary)) as Median_Salary from countryDF7 group by country order by Median_Salary desc")

# Here I add a dollar sign and a comma to the table in order to make it more presentable
spark.sql("create or replace temp view salaryByCountryDF as select Country, if(length(median_salary)=8,concat(left(concat('$',ceiling(Median_Salary)),4),',',right(concat('$',ceiling(Median_Salary)),3)),if(length(median_salary)=7,concat(left(concat('$',ceiling(Median_Salary)),3),',',right(concat('$',ceiling(Median_Salary)),3)),concat(left(concat('$',ceiling(Median_Salary)),2),',',right(concat('$',ceiling(Median_Salary)),3)))) as Median_Salary from countryDF8")

# COMMAND ----------

# MAGIC %md
# MAGIC **According to the data given, the country with the highest compensation based on it's median salary is the United States with a median salary of $105,000.**

# COMMAND ----------

# This table displays the country and it's corresponding median salary
display(spark.sql("select * from salaryByCountryDF"))

# COMMAND ----------

# MAGIC %md
# MAGIC The table above shows that there is a disparity between countries concerning salary for these types of STEM field jobs in particular. Based off the results from this analysis, the best country to work in with one of these types of jobs is the United States with a median salary of $105,000, with Switzerland coming in second with a median salary of $104,338. The worst country for someone to work in with one of these types of jobs based solely on median salary is Nigeria with a median salary of $1,812 and then Egypt coming in second with a median salary of $4,380. It is shocking to see the United States median salary is over 50 times more than Nigeria's median salary. These results may be partially due to the cost of living in these countries. Overall however, if someone held one of these types of jobs and worked in countries such as Nigeria or Egypt, it would benefit them greatly by taking their experience and working in other countries with a higher median salary instead such as the United States or Switzerland. 

# COMMAND ----------

# MAGIC %md
# MAGIC #Conclusion
# MAGIC In conclusion, based off the results from this analysis I have shown disparities in not only gender, but also in age, job, and country. 
# MAGIC 
# MAGIC Here are some of the key statistics from each of the analyzations I performed:
# MAGIC 
# MAGIC When analyzing gender split among the participants by country I found that each and every country had the male gender group outnumber all the other gender groups combined. This showed that jobs in the STEM field have a disproportionate amount of males compared to other genders internationally.
# MAGIC     
# MAGIC When analyzing differences in average and median salary among the different gender groups I found the male gender groups earned significantly more on average than females and nonbinaries as well as having a higher median salary than these gender groups as well. People with a different identity had both the highest average and median salary however this could be attributed to the lack of data for this gender group since there were only 27 rows with this specific gender.
# MAGIC 
# MAGIC When analyzing disparity in age concerning salary I found that salary and age have a positive relationship with each other, with a correlation of approximately .11. This means that as age increases, salary is likely to increase as well. I also created a table which showed the average salary per decade and found this to be true as well, as the average salary increased for each decade up until the 40s where it starts to decrease slightly. I found the average salary to peak for people in their 70s with an average salary of $190,000 although this is most likely the result of an outlier. The lowest average salary was for people in their 10s with an average salary of $50,000.
# MAGIC 
# MAGIC When looking for the job with the highest median salary I found that to be the Operations Research Practitioner with a median salary of $106,921. I also found the job with the lowest median salary to be a programmer with a median salary of $23,753. This analysis was useful because all the different salaries for jobs in these STEM fields. For example, for jobs concerning data, Data Scientist had the highest median salary of $68,343 while Data Miner had the lowest median salary with $36,547. Therefore based off the results from this data, it would benefit the Data Miner to use their experience to try to get a job as a Data Scientist.
# MAGIC 
# MAGIC In my final analysis I looked for disparities between countries concerning salary. I found the countries with the highest median salary to be the United States and Switzerland, with median salaries of $105,000 and $104,338 respectively. On the other hand, I found the countries with the lowest median salary to be Nigeria with $1,812 and Egypt with $4,380. Out of all the disparities I analyzed, I found the most shocking one to be that the United States had a median salary over 50 times larger than Nigeria. Thus, it can be deduced from the results of the analysis that it would be very beneficial to people who work in Nigeria, Egypt or any other country with a low median salary to use their experience in countries with a higher median salary such as in the United States or Switzerland. 
# MAGIC 
# MAGIC Therefore, I have shown and proven disparities in gender, age, job, and country concerning these types of jobs in the STEM field. Some of these disparities are warranted such as disparities in age concerning salary since age is an indicator of experience so the more experience one person has, naturally the higher their salary should be. However, other disparities are not warranted such as disparities in gender concerning salary, the fact that males outnumber all other gender groups internationally, and median salaries in some countries being many times more than median salaries in other countries. Efforts should be made to invite other gender groups to come into these types of fields and increase the salary for these gender groups as well. Efforts should also be made to increase salaries in countries with a low median salary such as in Nigeria or Egypt. 
