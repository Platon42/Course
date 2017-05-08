import pandas as pn

df = pn.read_csv('samples/titanic.csv')

def proc_freq(data,column):
    for c in column:
        x = data[c].value_counts().to_frame()
        nums = x.columns.tolist()[0]
        x.rename(columns={nums: 'Freq'}, inplace=True)
        x['Pct'] = x['Freq']/x.Freq.sum()
        x['Freq Acum'], x['Pct Acum'] = x.Freq.cumsum(),x.Pct.cumsum()
        x.sort_values(['Freq'],ascending=[0],inplace=True)
    print('Frequency distribution table for the variable '+nums)
    print('\n')
    print(x)
    print('\n')

female = df[df['Sex'] == 'female']
female['ClearName'] = female.Name.str.extract('\.(\s\w[a-z]*)')

male = df['Sex'] == 'male'

male_count = male.value_counts()[1]
female_count = male.value_counts()[0]

all_data_survive = df['Survived']

not_survive = all_data_survive.value_counts()[0]
survive = all_data_survive.value_counts()[1]
total_survive = survive + not_survive

first_class = df['Pclass'] == 1

v_fclass = first_class.value_counts()[1]
v_oclass = first_class.value_counts()[0]
total_class = v_fclass + v_oclass

mean_age = df["Age"].mean()
median_age = df["Age"].median()
corr = df['SibSp'].corr(df['Parch'])


print("Pearson correlation is", corr)
print("Mean age =", mean_age, "Median age =", median_age)
print("Male -", male_count, "Female -", female_count)
print("Survive percent =", (float(survive) / total_survive * 100))
print("First class passengers =", (float(v_fclass) / total_class * 100))
proc_freq(female, ['ClearName'])