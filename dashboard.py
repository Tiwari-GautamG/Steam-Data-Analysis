import panel as pn
from sqlalchemy import create_engine
import pandas as pd
import hvplot.pandas
import plotly.graph_objects as go
from bokeh.models import PrintfTickFormatter
import numpy as np


db_username = 'root'
db_password = 'root'
db_host = 'localhost'
db_name = 'steam_data'
engine = create_engine(f'mysql+mysqlconnector://{db_username}:{db_password}@{db_host}/{db_name}')
query = 'select * from steamout'
df = pd.read_sql(query, engine)
pn.extension('plotly')


def create_hist(feature):
    if feature == 'Price':
        return df[df['Price'] < 5000].hvplot.hist(y='Price', bins=10, hue='Reviews')
    else:
        return df.hvplot.hist(y='Discount', bins=10, color='orange')


def create_pie_chart(x_axis):
    val, label = None, None
    if x_axis == 'DLC':
        val = df['DLC'].value_counts()
        label = list(df['DLC'].unique())

    if x_axis == 'Tags':
        top_10tgs = df.groupby('Tag1')['Total_revenue'].sum().reset_index().sort_values(by='Total_revenue')[-10:]
        val = top_10tgs['Total_revenue']
        label = list(top_10tgs['Tag1'])

    fig = go.Figure(data=[go.Pie(labels=label, values=val)])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='white')),
        width=500,
        height=500
    )
    pie_chart_pane = pn.pane.Plotly(fig, sizing_mode='stretch_width')
    return pie_chart_pane


def y_formatter(x, pos):
    return '{:,.0f}'.format(x)


def create_bar_chart(sel):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#00FFFF", "#FF00FF", "#32CD32"]
    dfb, x, y, yl = None, None, None, None
    if sel == 'Discount by Reviews':
        dfb = df.groupby('Reviews')['Discount'].sum().reset_index()
        x = 'Reviews'
        y = 'Discount'
        yl = 'Discount'

    if sel == 'Positivity by DLC':
        dfb = df.groupby('DLC')['Positivity_Percentage'].sum().reset_index()
        y = 'Positivity_Percentage'
        x = 'DLC'
        yl = 'Positivity'

    dfb['Color'] = colors[:len(dfb)]
    return dfb.hvplot.bar(x=x, y=y, color='Color', rot=90, width=450, height=300, ylabel=yl).opts(yticks=5, yformatter=PrintfTickFormatter(format='%0.2f'))

def top_n_games(no):
    sorteddf = df.sort_values(['Normalized'], ascending=False)
    sorteddf = sorteddf.loc[sorteddf['Year'] > 2003]
    return pn.pane.DataFrame(sorteddf.groupby('Year')[['Title', 'Year']].head(no).sort_values(by='Year'), height=450, width=500, index=False)


# https://tabler-icons.io/
button1 = pn.widgets.Button(name="Introduction", button_type="warning", icon="info-circle", styles={"width": "100%"})
button2 = pn.widgets.Button(name="Games", button_type="warning", icon="clipboard-data", styles={"width": "100%"})
button3 = pn.widgets.Button(name="Distribution", button_type="warning", icon="chart-histogram", styles={"width": "100%"})
button4 = pn.widgets.Button(name="Piecharts", button_type="warning", icon="chart-pie", styles={"width": "100%"})
button5 = pn.widgets.Button(name="Bar charts", button_type="warning", icon="chart-bar", styles={"width": "100%"})
button6 = pn.widgets.Button(name="Game Info", button_type="warning", icon="device-gamepad-2", styles={"width": "100%"})

dist_feature = pn.widgets.Select(name="Dist", options=['Price', 'Discount'])
no_of_tops = pn.widgets.Select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
pie_features = pn.widgets.Select(name='Feature', options=['DLC', 'Tags'])
x_bar = pn.widgets.Select(name='Feature', options=['Positivity by DLC', 'Discount by Reviews'])

textbox = pn.widgets.TextInput(placeholder='Search')
but = pn.widgets.Button(name='Go')
sel = pn.widgets.Select()


n = pn.pane.HTML(width=300)
r = pn.pane.HTML(width=300)
t = pn.pane.HTML(width=300)
p = pn.pane.HTML(width=300)
global txt
img = pn.pane.Image(width=300)
rhead = pn.pane.HTML(width=300)


def clik(event):
    df1 = df.copy()
    val = textbox.value
    res = df1.loc[df['Title'].str.lower().str.contains(val)][['Title']].sort_values(by='Title')['Title'].to_list()
    sel.options = res


but.on_click(clik)


def update(event):
    tt = ""
    val = event.new

    main_val = df.loc[df['Title'] == val]
    r.object = f"<span style='color:red;font-size:14px;'><b>Review - </b></span> {main_val['Reviews'].values[0]}"
    n.object = f"<span style='color:red;font-size:14px;'><b>Name - </b></span> {main_val['Title'].values[0]}"
    t1 = main_val['Tag1'].values[0]
    t2 = main_val['Tag2'].values[0]
    t.object = f"<span style='color:red;font-size:14px;'><b>Tags - </b></span> {t1 + ', ' + t2}"
    p.object = f"<span style='color:red;font-size:14px;'><b>Price - </b></span> ₹{main_val['Price'].values[0]}"
    if main_val['Price'].values[0] == 0:
        p.object = f"<span style='color:red;font-size:14px;'><b>Price - </b></span> Free"

    img.object = main_val['Image'].values[0]
    rhead.object = "<span style='color:blue;font-size:24px;'><b>Recommendations</b> </span>"
    rec_df = df[(df['Tag1'] == t1) & (df['Tag2'] == t2)]
    top_recc = rec_df.sort_values(by='Normalized', ascending=False)[:5]['Title'].values

    if val in top_recc:
        index_to_pop = np.where(top_recc == val)[0][0]
        top_recc = np.delete(top_recc, index_to_pop)

    for i in top_recc:
        imag = df[df['Title'] == i]['Image'].values[0]
        tt += f"""
#### {i} <br>

<img src="{imag}" alt="Alt Text" width="200" height="150"><br>
"""
    markdown_pane.object = tt


sel.param.watch(update, 'value')
markdown_pane = pn.pane.Markdown("")

def show_page(page_key):
    main_area.clear()
    main_area.append(mapping[page_key])

button1.on_click(lambda event: show_page("Page1"))
button2.on_click(lambda event: show_page("Page2"))
button3.on_click(lambda event: show_page("Page3"))
button4.on_click(lambda event: show_page("Page4"))
button5.on_click(lambda event: show_page("Page5"))
button6.on_click(lambda event: show_page("Page6"))


def CreatePage1():
    descr = '''
    # Game Dataset Overview

    This dataset contains detailed information about **video games** available on a popular digital distribution platform. It offers insights into the characteristics and performance metrics of various games, providing valuable information for analysis.

    ## Dataset Attributes

    - **Title**: The name of the game.
    - **RAM**: The amount of RAM required to run the game (in MB).
    - **Size**: The size of the game file (in MB).
    - **Discount**: The discount applied to the game (in percentage).
    - **Price**: The retail price of the game (₹).
    - **DLC**: Indicates whether downloadable content (DLC) is available for the game (Yes/No).
    - **Tag1**: The primary genre or category of the game (e.g., JRPG, Simulation).
    - **Tag2**: A secondary genre or category that further describes the game (e.g., Story Rich, Flight).
    - **Year**: The release year of the game.
    - **ReviewNum**: The number of reviews the game has received.
    - **Positivity_Percentage**: The percentage of positive reviews.
    - **Normalized**: A normalized score reflecting the overall rating or performance of the game.
    - **Reviews**: The sentiment of the reviews (Positive/Mixed/Negative).
    - **Total_revenue**: The total revenue generated by the game.
    - **Image**: Link to the icon of the game
    '''

    return pn.Column(pn.pane.Markdown(descr, width=550),
                     align="center")

def CreatePage2(topg):
    return pn.Column(
        pn.pane.Markdown("## Top N Games"),
        pn.Row(no_of_tops, pn.bind(top_n_games, no_of_tops)),
        align="center"
    )

def CreatePage3():
    return pn.Column(
        pn.pane.Markdown("## Explore Distribution of Features"),
        dist_feature,
        pn.bind(create_hist, dist_feature),
        align="center",
    )

def CreatePage4():
    return pn.Column(
        pn.pane.Markdown("## Proportion of Feature values"),
        pie_features,
        pn.bind(create_pie_chart, pie_features),
        align="start",
    )


def CreatePage5():
    return pn.Column(
        pn.pane.Markdown("## Various Bar charts"),
        x_bar,
        pn.bind(create_bar_chart, x_bar),
        align="center",
    )


def CreatePage6():
    return pn.Column(pn.Column(pn.Row(textbox, but), sel, img, n, r, t, p, pn.Row(rhead, align='center'), markdown_pane, align='start'))


mapping = {
    "Page1": CreatePage1(),
    "Page2": CreatePage2(no_of_tops),
    "Page3": CreatePage3(),
    "Page4": CreatePage4(),
    "Page5": CreatePage5(),
    "Page6": CreatePage6()
}


sidebar = pn.Column(pn.pane.Markdown("## Pages"), button1, button2, button3, button4, button5, button6, styles={"width": "100%", "padding": "15px"})
main_area = pn.Column(mapping["Page1"], styles={"width": "100%"})

template = pn.template.BootstrapTemplate(
    title=" Game Data Analysis",
    sidebar=[sidebar],
    main=[main_area],
    header_background="black",
    theme=pn.template.DarkTheme,
    sidebar_width=200,
    busy_indicator=None,
)
template.servable()

