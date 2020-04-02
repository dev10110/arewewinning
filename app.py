import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests

from adjustText import adjust_text

st.title("Are we winning?")



###### Collect raw data

st.write("Reading data from https://github.com/CSSEGISandData/COVID-19")
repo = 'CSSEGISandData/COVID-19'

# raw url is the url to get the raw data
# requests_url is the url at which the api call to get the last commit date can be found.
raw_url ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
request_url = 'https://api.github.com/repos/{0}/commits?path={1}'

##########
st.subheader('Getting global confirmed cases')

path_confirmed = 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv';
df_confirmed = pd.read_csv(raw_url + path_confirmed)

if st.checkbox('Show Confirmed Raw Data'):
    st.write(df_confirmed);

try:
    r = requests.get(request_url.format(repo, path_confirmed))
    date_updated = r.json()[0]['commit']['committer']['date']
    st.write('Last Updated ' + date_updated)
except:
    pass
##########
st.subheader('Getting global reported deaths')
path_deaths = 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
df_deaths = pd.read_csv(raw_url + path_deaths)

if st.checkbox('Show Deaths (Raw Data)'):
    st.write(df_deaths);

try:
    r = requests.get(request_url.format(repo, path_deaths))
    date_updated = r.json()[0]['commit']['committer']['date']
    st.write('Last Updated ' + date_updated)
except:
    pass


##########
st.subheader('Getting global recovered cases')
path_recovered = 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
df_recovered = pd.read_csv(raw_url + path_recovered)

if st.checkbox('Show Recovered (Raw Data)'):
    st.write(df_recovered);

try:
    r = requests.get(request_url.format(repo, path_recovered))
    date_updated = r.json()[0]['commit']['committer']['date']
    st.write('Last Updated ' + date_updated)
except:
    pass

#### Raw Data Loaded

#### Corrections to data
st.subheader("Process the data...")
st.write('Note, the data from Guyana is removed as it seems weird')
# remove lat long data
df_confirmed2=df_confirmed.drop(columns=['Lat','Long']).groupby('Country/Region').sum().transpose().drop(columns=['Guyana'])
df_deaths2=df_deaths.drop(columns=['Lat','Long']).groupby('Country/Region').sum().transpose().drop(columns=['Guyana'])
df_recovered2=df_recovered.drop(columns=['Lat','Long']).groupby('Country/Region').sum().transpose().drop(columns=['Guyana'])

# make index into timeseries
df_confirmed2.index = pd.to_datetime(df_confirmed2.index)
df_deaths2.index = pd.to_datetime(df_deaths2.index)
df_recovered2.index = pd.to_datetime(df_recovered2.index)

# add world
df_confirmed2['World'] = df_confirmed2.sum(axis=1)
df_deaths2['World'] = df_deaths2.sum(axis=1)
df_recovered2['World'] = df_recovered2.sum(axis=1)


if st.checkbox('Show Processed Data'):
    st.write('Confirmed Cases:')
    st.write(df_confirmed2);
    st.write('Deaths Cases:')
    st.write(df_deaths2);
    st.write('Recovered Cases:')
    st.write(df_recovered2);



st.subheader('Plot')

##### collect information on new cases
window = st.slider("Sum new cases over how many days?", min_value=1, max_value=15, value=3, step=1) # days

# number of new cases that will be created today
df_confirmed_new = df_confirmed2.diff().rolling(f'{window}d').sum()
df_deaths_new = df_deaths2.diff().rolling(f'{window}d').sum()
df_recovered_new = df_recovered2.diff().rolling(f'{window}d').sum()




xopts = ["Total Confirmed Cases", "Total Deaths", "Total Recovered"]
yopts = ["New Confirmed Cases", "New Deaths", "New Recovered"]

xaxis = st.radio('X-Axis Type', xopts, index=0)
yaxis = st.radio('Y-Axis Type', yopts, index=0)

highlight = st.multiselect("Select the countries to highlight", list(df_confirmed2.columns), default=["World", "US", "Italy", "China", "Korea, South", "Singapore"])

if xaxis == 'Total Confirmed Cases':
    x = df_confirmed2
if xaxis == 'Total Deaths':
    x = df_deaths2
if xaxis == 'Total Recovered':
    x = df_recovered2


if yaxis == 'New Confirmed Cases':
    y = df_confirmed_new
if yaxis == 'New Deaths':
    y = df_deaths_new
if yaxis == 'New Recovered':
    y = df_recovered_new


# start plotting
fig = plt.figure()

if len(highlight) > 0:
    for c in df_confirmed2.columns:
        if not c in highlight:
            plt.plot(x[c],y[c], 'gray', label=c)
    for h in highlight:
        plt.plot(x[h],y[h], label=h)
else:
    for c in df_confirmed2.columns:
            plt.plot(x[c],y[c], label=c)


handles, labels = plt.gca().get_legend_handles_labels()
# sort both labels and handles by total number of cases
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: y[t[0]][-1], reverse=True))

# only show the top 10 countries in the labels
labels2=[labels[i] for i in range(len(labels)) if i < 10 or labels[i] in highlight]
handles2=[handles[i] for i in range(len(handles)) if i < 10 or labels[i] in highlight]

#plt.gca().legend(handles2, labels2,loc='upper left', bbox_to_anchor= (1.01, 1), ncol=1,
#            borderaxespad=0, frameon=True)

plt.yscale('log')
plt.xscale('log')

plt.xlabel(xaxis)
plt.ylabel(f'{yaxis} over last {window} days')

plt.grid(True)
plt.xlim(left=50)
plt.ylim(bottom=10)

# plot the reference line
factor=0.5


xs = np.logspace(0,6)
ys = xs
plt.plot(xs, ys,'k--')

# collect all the force points
allx, ally = [],[]

for c in df_confirmed2.columns:
        allx.extend(x[c])
        ally.extend(y[c])

# create the annocations
texts = []

# Rotate angle
pt = np.array((500, 500))
trans_angle = plt.gca().transData.transform_angles(np.array((45,)),
                                                   pt.reshape((1, 2)))[0]
plt.text(500,600, '1 new cases for every total case', rotation=trans_angle, ha='left', va='bottom')


for l in highlight:
    texts.append(plt.text(x[l][-1],y[l][-1],l))


if len(texts)>0:
    adjust_text(texts, x=allx, y=ally, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), force_text=(1, 2.5), force_points=(2, 20), only_move={'points':'y', 'text':'y', 'objects':'y'})

st.pyplot()

st.markdown(f"""
## How to read this plot?

On the x axis we have the total number of cases, on a log scale. Every division has 10 times as many cases.

On the y axis we have the incremental number of cases - its a sum of the cases over the last {window} days.

There is no time on this axis - time generally increases to up and to the right.
But by removing this information, we can see that most countries follow a very similar path - they all keep growing up and to the right.

**However, if a country is doing well,** it will peel away from this trend, and show drop.
This means that now, even though they have a large number of cases, the number of new cases is dropping!

The next plot shows this growth rate (number of new cases during the window/total number of cases)
""")


st.subheader("So who's winning?")

window2 = st.slider("Over how many days?", min_value=1, max_value=15, value=7, step=1) # days



st.write(f'Lets plot the fraction of total cases that happened in the last {window2} days')
st.write('But lets filter out places that have had less than 100 cases in total.')

# number of new cases that will be created today
df_confirmed_new = df_confirmed2.diff().rolling(f'{window2}d').sum()
df_deaths_new = df_deaths2.diff().rolling(f'{window2}d').sum()
df_recovered_new = df_recovered2.diff().rolling(f'{window2}d').sum()


df_ratio_nc_c = (df_confirmed_new/df_confirmed2)

# mask the ones that have had less than 100 cases
df_ratio_nc_c.mask(df_confirmed2<100, other=np.nan, inplace=True, axis=None, level=None, errors='raise', try_cast=False, raise_on_error=None)

highlight2 = st.multiselect("Select countries to highlight", list(df_ratio_nc_c.columns), default=["World", "US", "Italy", "China", "Korea, South", "Singapore", "Diamond Princess"])


plt.figure()
if len(highlight2) > 0:
    ax = df_ratio_nc_c.plot(legend=False, color='gray')
    df_ratio_nc_c[highlight2].plot(ax=ax)
else:
    df_ratio_nc_c.plot()
plt.xlabel('Date')
plt.title(f'Fraction of cases that occured in the last {window2} days')
plt.ylabel('Fraction')
st.pyplot()

st.write('We can clearly see China ahead of the countries')


st.write('If we zoom in on more recent times, (the last two months)')
plt.figure()
if len(highlight2) > 0:
    ax = df_ratio_nc_c.plot(legend=False, color='gray')
    df_ratio_nc_c[highlight2].plot(ax=ax)
else:
    df_ratio_nc_c.plot()
xlim = plt.xlim()
plt.xlim(left = xlim[1]-60)
plt.xlabel('Date')
plt.title(f'Fraction of cases that occured in the last {window2} days')
plt.ylabel('Fraction')
st.pyplot()

st.write('We can see an encouraging downwards trend. ')


# see the latest data
df_ratio_nc_c_last = df_ratio_nc_c.iloc[-1].dropna()
ranks = df_ratio_nc_c_last.rank(method='max')

df = pd.concat([df_ratio_nc_c_last, ranks], axis=1)
df.columns = ["Fraction", "Rank"]

df = df.sort_values(by='Rank')

# add the top and bottom ten to the highlight list
highlight3 = highlight2.copy(); # save for bold
highlight2.extend(df[:10].index.to_list())
highlight2.extend(df[-10:].index.to_list())

filtered_df = df[df.index.isin(highlight2)]

filtered_df = filtered_df.reset_index().set_index("Rank", drop=False) # make rank the main index
df = df.reset_index().set_index("Rank", drop=False) # make rank the main index

st.write('And as of the latest datapoint, this is the fraction for each country')


def highlight_list(x):
    if x['Country/Region'] in highlight3:
        return ['background-color: darkorange']*3
    else:
        return ['background-color: white']*3

if st.checkbox('Show Full Ranking'):
    st.dataframe(df.style.apply(highlight_list, axis=1))
else:
    st.dataframe(filtered_df.style.apply(highlight_list, axis=1))


st.write('Visit my github (https://github.com/dev10110/arewewinning) to see the code behind this, and let me know if there is anything else you want to see')

#
