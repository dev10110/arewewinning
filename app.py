import streamlit as st

import numpy as np
import pandas as pd

import plotly.figure_factory as ff
import plotly as py
import plotly.graph_objects as go

import requests
import warnings

from adjustText import adjust_text


st.title("Are we winning?")

st.markdown(r"""


We will plot the data in a style inspired by MinutePhysics @ https://www.youtube.com/watch?v=54XLXg4fYsc

Essentially, with exponential curves its very difficult to predict whether you've started to hit the plateau.
So, instead of plotting things on the time axis, we choose to plot them in such a way that the exponential growth is highlighted.

If a curve $$N(t)$$ is exponential, it follows this rule:

$$\frac{dN}{dt} = k N$$

which means that the growth rate is proportional to the current count. In the case of modelling pandemics, a slower pandemic is when this growth rate decreases.
If its 0 that means that you have no additional cases, and essentially means you've won.

So lets find this constant for various countries. We will plot the increase in number of cases (which is proportional to $$\frac{dN}{dt}$$) against the total number of cases thus far (a proxy for the current number of infected people).

Really, this ignores most of the detailed modelling (like the SEIR model, or contact-dynamics based models) that exist.

It just provides a different perspective to everyday's news. Instead of being surprised by the record high number of cases everyday, we should look out for slowing growth rates, and hopefully this applet helps you see it for yourself.

Play around with the data, and if you have any suggestions for more stuff I should add, please fork the project or let me know!

""")

# Collect raw data
st.header("Gathering Data")

st.write("Reading data from https://github.com/CSSEGISandData/COVID-19")
repo = 'CSSEGISandData/COVID-19'

# raw url is the url to get the raw data
# requests_url is the url at which the api call to get the last commit date can be found.
raw_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/'
request_url = 'https://api.github.com/repos/{0}/commits?path={1}'

# confirmed
st.subheader('Getting global confirmed cases')

path_confirmed = 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df_confirmed = pd.read_csv(raw_url + path_confirmed)

if st.checkbox('Show Confirmed Raw Data'):
    st.write(df_confirmed)

try:
    r = requests.get(request_url.format(repo, path_confirmed))
    date_updated = r.json()[0]['commit']['committer']['date']
    st.write('Last Updated ' + date_updated)
except:
    pass


# deaths
st.subheader('Getting global reported deaths')
path_deaths = 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
df_deaths = pd.read_csv(raw_url + path_deaths)

if st.checkbox('Show Deaths (Raw Data)'):
    st.write(df_deaths)

try:
    r = requests.get(request_url.format(repo, path_deaths))
    date_updated = r.json()[0]['commit']['committer']['date']
    st.write('Last Updated ' + date_updated)
except:
    pass


# recovered
st.subheader('Getting global recovered cases')
path_recovered = 'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
df_recovered = pd.read_csv(raw_url + path_recovered)

if st.checkbox('Show Recovered (Raw Data)'):
    st.write(df_recovered)

try:
    r = requests.get(request_url.format(repo, path_recovered))
    date_updated = r.json()[0]['commit']['committer']['date']
    st.write('Last Updated ' + date_updated)
except:
    pass

# Raw Data Loaded

# Corrections to data
st.subheader("Process the data...")
st.write("Note, the data from Guyana is removed as it's data seems weird")

# remove lat long data
df_confirmed2 = df_confirmed.drop(columns=['Lat', 'Long']).groupby(
    'Country/Region').sum().transpose().drop(columns=['Guyana'])

df_deaths2 = df_deaths.drop(columns=['Lat', 'Long']).groupby(
    'Country/Region').sum().transpose().drop(columns=['Guyana'])

df_recovered2 = df_recovered.drop(columns=['Lat', 'Long']).groupby(
    'Country/Region').sum().transpose().drop(columns=['Guyana'])

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
    st.write(df_confirmed2)
    st.write('Deaths Cases:')
    st.write(df_deaths2)
    st.write('Recovered Cases:')
    st.write(df_recovered2)


st.header('Plot')
########

#country = 'Singapore'

#df_death_perc = 100*(df_deaths2[country]/df_confirmed2[country])
#df_recov_perc = 100*(df_recovered2[country]/df_confirmed2[country])
#df_death_perc.dropna()
#df_recov_perc.dropna()

#st.area_chart(df_death_perc )

#st.pyplot()

#plt.figure();



#st.pyplot()

#df_recovered2



########

# collect information on new cases
window = st.slider("Sum new cases over how many days?", min_value=1,
                   max_value=15, value=3, step=1)  # days

xopts = ["Total Confirmed Cases", "Total Deaths",
         "Total Recovered", "Total (Confirmed - Recovered)"]
yopts = ["New Confirmed Cases", "New Deaths", "New Recovered"]

xaxis = st.radio('X-Axis Type', xopts, index=3)
yaxis = st.radio('Y-Axis Type', yopts, index=0)

highlight = st.multiselect("Select the countries to highlight", list(df_confirmed2.columns), default=[
                           "World", "US", "Italy", "China", "Korea, South", "Singapore"])

factor = st.slider("Growth constant", min_value=0.0,
                   max_value=2.0, value=1.0, step=0.1)


# number of new cases that will be created today, and applying a rolling sum filter
df_confirmed_new = df_confirmed2.diff().rolling(f'{window}d').sum()
df_deaths_new = df_deaths2.diff().rolling(f'{window}d').sum()
df_recovered_new = df_recovered2.diff().rolling(f'{window}d').sum()


if xaxis == 'Total Confirmed Cases':
    x = df_confirmed2
if xaxis == 'Total Deaths':
    x = df_deaths2
if xaxis == 'Total Recovered':
    x = df_recovered2
if xaxis == 'Total (Confirmed - Recovered)':
    x = df_confirmed2 - df_recovered2


if yaxis == 'New Confirmed Cases':
    y = df_confirmed_new
if yaxis == 'New Deaths':
    y = df_deaths_new
if yaxis == 'New Recovered':
    y = df_recovered_new

# get the date as a string
datestring = x.index.strftime("%Y-%m-%d")


data = []
# if you dont choose any to highlight, all will be colored

# plot everything in gray if the length of highlight is 0
if len(highlight) == 0:
    for c in df_confirmed2.columns:
        data.append(go.Scattergl(x=x[c], y=y[c], name=c, showlegend=False, hovertext=datestring))
        #plt.plot(x[c], y[c], label=c)
else:
    for c in df_confirmed2.columns:
        if not c in highlight:
            data.append(go.Scattergl(x=x[c], y=y[c], name=c, line=dict(color='gray'), showlegend=False, hovertext=datestring))
        #plt.plot(x[c], y[c], 'gray', label=c)

# plot all the highlighted ones on top
for h in highlight:
    data.append(go.Scattergl(x=x[h], y=y[h], name=h, hovertext=datestring))
    #plt.plot(x[h], y[h], label=h)


xrange=np.logspace(1,7)
data.append(go.Scattergl(x=xrange, y = factor*xrange, name='Growth Constant', line=dict(color='black', dash="dash"), hovertext=f'{factor*100:.0f} new cases for every 100 cases'))


layout = go.Layout(xaxis_type="log", yaxis_type="log", legend={'traceorder':'normal'}, xaxis = dict(range=(2,7)), yaxis = dict(range=(1,6)),
    xaxis_title=xaxis,
    yaxis_title=f'{yaxis} over last {window} days')

fig = go.Figure(data, layout)

st.plotly_chart(fig)


st.markdown(f"""
## How to read this plot?

On the x axis we have the total number of cases, on a log scale. Every division has 10 times as many cases.

On the y axis we have the incremental number of cases - its a sum of the cases over the last {window} days.

There is no time on this axis - time generally increases to up and to the right.
But by removing this information, we can see that most countries follow a very similar path - they all keep growing up and to the right.

**However, if a country is doing well,** it will peel away from this trend, and show drop.
This means that now, even though they have a large number of cases, the number of new cases is dropping!

In fact, if you plot Total (Confirmed - Recovered) on the x axis, you can see China (and a few other countries) walk backwards, down and to the left.
This means that more people are recovering rather than becoming infected.

The next plot shows this growth rate (number of new cases during the window/total number of cases)
""")


st.subheader("So who's winning?")


st.write(f'Lets plot the fraction of total cases that happened in the last some number of days')

window2 = st.slider("Over how many days?", min_value=1, max_value=15, value=7, step=1)  # days

st.write('But lets filter out places that have had less than 100 cases in total.')

# number of new cases that will be created today
df_confirmed_new = df_confirmed2.diff().rolling(f'{window2}d').sum()
df_deaths_new = df_deaths2.diff().rolling(f'{window2}d').sum()
df_recovered_new = df_recovered2.diff().rolling(f'{window2}d').sum()


df_ratio = (df_confirmed_new/df_confirmed2)

# mask the ones that have had less than 100 cases
df_ratio.mask(df_confirmed2 < 100, other=np.nan, inplace=True)

highlight2 = st.multiselect("Select countries to highlight", list(df_ratio.columns), default=[
                            "World", "US", "Italy", "China", "Korea, South", "Singapore", "Diamond Princess"])

data2 = []
if len(highlight2) > 0:
    for c in df_ratio.columns:
        if not c in highlight2:
            data2.append(go.Scattergl(x = df_ratio[c].index, y=df_ratio[c], name=c, line=dict(color='gray'), showlegend=False))
        else:
            data2.append(go.Scattergl(x = df_ratio[c].index, y=df_ratio[c], name=c, showlegend=True))
    #ax = df_ratio.plot(legend=False, color='gray')
    #df_ratio[highlight2].plot(ax=ax)
else:
    for c in df_ratio.columns:
        data2.append(go.Scattergl(x = df_ratio[c].index, y=df_ratio[c], name=c, showlegend=False))

layout2 = go.Layout(title=f'Fraction of cases that occured in the last {window2} days', xaxis_title='Date',
    yaxis_title='Fraction')

fig2 = go.Figure(data2, layout2)

st.plotly_chart(fig2)

st.write('We can clearly see China ahead of the countries, but also an encouraging downwards trend.')


# see the latest data
df_ratio_latest = df_ratio.iloc[-1].dropna()
ranks = df_ratio_latest.rank(method='max')

# create a summary data frame
df_summary = pd.concat([df_ratio_latest, ranks], axis=1)
df_summary.columns = ["Fraction", "Rank"]

df_summary = df_summary.sort_values(by='Rank')

# add the top and bottom ten to the highlight list
highlight3 = highlight2.copy()  # save for bold
highlight2.extend(df_summary[:10].index.to_list())
highlight2.extend(df_summary[-10:].index.to_list())


# make a summary with the top 10, bottom 10 and the few that were highlighted
filtered_df = df_summary[df_summary.index.isin(highlight2)]
filtered_df = filtered_df.reset_index().set_index("Rank", drop=False)  # make rank the main index
df_summary = df_summary.reset_index().set_index("Rank", drop=False)  # make rank the main index

st.write('And as of the latest datapoint, this is the growth rate for each country')


def highlight_list(x):
    if x['Country/Region'] in highlight3:
        return ['background-color: darkorange']*3
    else:
        return ['background-color: white']*3


if st.checkbox('Show Full Ranking'):
    st.dataframe(df_summary.style.apply(highlight_list, axis=1))
else:
    st.dataframe(filtered_df.style.apply(highlight_list, axis=1))


st.markdown("""

So. What does this mean?
Well, everyday the graphs will be different, and in fact the settings you choose can show you optimistic or pessimistic views, as you desire.

But overall, I'm seeing trends of lower growth rates. And China's recovery shows that they were not wrong in saying that COVID-19 can be controlled.
If the world woke up a bit sooner and put in harsher controls, and provided the healthcare resources needed earlier, perhaps the impact wouldn't have been as serious.

But today, (April 2 2020), the data shows me that things are getting better.


""")


st.video('https://www.youtube.com/watch?v=54XLXg4fYsc')



st.header("Are people recovering?")

st.markdown(r"""At the suggestion of a friend, it seemed interesting to also plot the percentage of recovery and percentage of infections.

There are a few different plots I've seen along these lines:

1) Plot the percentage of people that have recovered and died against time. In this graph you plot:

$$ \frac{\text{Total Recovered so far}}{\text{Total Recovered so far + Total dead so far}} $$
 and
$$ \frac{\text{Total Dead so far}}{\text{Total Recovered so far + Total dead so far}} $$

on the same graph, so they add to 100%. However this seems to ignore an important time factor here: you only know that a person is recovered many days after they can be confirmed as a case.
Also, seeing the split between recovered and dead doesnt hightlight anything about the current cummulative survival rates as the number of people who are currently infected is missing in that data.

2) A population plot.

This is quite common in the modelling community, where you simulate the total population and see whether a sizeable fraction gets infected.
These consider how many of the people are 'removed' from the simulation, but most dont seem to make the split bewteen dead and recovered.
Either because its irrelevant to the model, or because it is hard to tell the difference.

So instead I'm going to plot percentages of infected people that have recovered or died. Its simply,
$$ \frac{\text{Total number of recoveries}}{\text{Total number of cases}} $$
and
$$ \frac{\text{Total number of deaths}}{\text{Total number of cases}} $$
stacked on top of each other, and I'll give you the option to see it normalized or as a quantity.

These graphs are supposed to resemble those of https://www.washingtonpost.com/graphics/2020/world/corona-simulator/
but beware that Harry Stevens uses the total population on the y axis, not just the total number of confirmed cases.

 """)


df_death_perc = (df_deaths2/df_confirmed2)
df_recov_perc = (df_recovered2/df_confirmed2)



layout3 = go.Layout(xaxis_title='Date',
    yaxis_title='All Cases', yaxis=dict(
    tickformat= ',.0%',
    range= [0,1]
  ))


highlight4 = st.multiselect("Select countries to highlight", list(df_death_perc.columns), default=[
                            "World", "US", "Italy", "China", "Singapore", "Diamond Princess", "Spain"])

for h in highlight4:
    data3 = []
    data3.append(go.Scatter(x=df_death_perc.index, y=list(df_death_perc[h]),fill='tozeroy', name=f'{h} (Death)', hovertext=df_deaths2[h]))
    data3.append(go.Scatter(x=df_death_perc.index, y=list(1-df_recov_perc[h]), name=f'{h} (Recovery)', hovertext=df_recovered2[h]))
    data3.append(go.Scatter(x=df_death_perc.index, y=list(1 for d in df_recov_perc[h]), fill='tonexty',name=f'{h} (Total)', hovertext=df_confirmed2[h]))

    fig3 = go.Figure(data3,layout3)
    fig3.update_layout(title=h)
    st.plotly_chart(fig3)



st.markdown("""
Interpreting these graphs is not trivial.

The few countries I have selected all show vastly different types of plots, but here are a few things to look out for:

1) If the blue + green nearly meet, that means almost all the confirmed cases have either recovered or died. That suggests that the country has a strong handle on the number of infections. (See China)

2) If the two regions started to meet but then grew away from each other, that indicates an increase in total number of confirmed cases, but who have neither recovered nor died (See US)

3) Steps in the graph are a telltale sign of poor data collection or very small total number of cases (See Diamond Princess and early US)

4) Spain's graph is just weird.


Visit my github (https://github.com/dev10110/arewewinning) to see the code behind this, and let me know if there are other plots you want to see.


Stay safe!
Dev
""")
#
