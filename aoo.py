import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Population Analysis Netherlands",
                   page_icon=":bar_chart:")

# Load the dataset
file_path = 'merged_population_data - Clean DB population.csv'  # Update the file path
data = pd.read_csv(file_path)

# Filter to exclude Dutch nationality if selected
exclude_dutch = st.checkbox('Exclude Dutch Nationality')

if exclude_dutch:
    data = data[data['Nationality'] != 'Dutch']

# Helper function to get total data per year
def get_total_data_per_year(data):
    return data.iloc[:, 3:].sum()

# Helper function to get data by continent
def get_continent_data(data):
    return data.groupby('Continent').sum().iloc[:, 1:]

# Helper function to get data by selected nationalities
def get_nationality_data(data, nationalities):
    return data[data['Nationality'].isin(nationalities)]

# Calculate percentage change
def calculate_percentage_change(start_value, end_value):
    if start_value == 0:
        return float('inf')
    return (end_value - start_value) / start_value

# Total population trends over the years
def plot_total_data():
    total_data_per_year = get_total_data_per_year(data)
    fig = px.line(total_data_per_year, x=total_data_per_year.index, y=total_data_per_year.values,
                  labels={'index': 'Year', 'y': 'Population'}, 
                  title='Population Trends in the Netherlands (2010-2023)')
    fig.update_traces(hovertemplate='Year: %{x}<br>Population: %{y:,}')
    st.plotly_chart(fig)

# Population trends by continent
def plot_continent_data():
    continent_data = get_continent_data(data)
    continent_data = continent_data.T  # Transpose to have years as rows and continents as columns
    fig = go.Figure()
    for continent in continent_data.columns:
        fig.add_trace(go.Scatter(x=continent_data.index, y=continent_data[continent], mode='lines+markers', name=continent,
                                 hovertemplate='Year: %{x}<br>Population: %{y:,}'))
    fig.update_layout(title='Population Trends by Continent (2010-2023)', 
                      xaxis_title='Year', yaxis_title='Population')
    st.plotly_chart(fig)

# Population trends by selected nationalities
def plot_nationality_data(nationalities):
    nationality_data = get_nationality_data(data, nationalities)
    fig = go.Figure()
    for nationality in nationalities:
        fig.add_trace(go.Scatter(x=nationality_data.columns[3:], 
                                 y=nationality_data[nationality_data['Nationality'] == nationality].iloc[0, 3:], 
                                 mode='lines+markers', name=nationality,
                                 hovertemplate='Year: %{x}<br>Population: %{y:,}'))
    fig.update_layout(title='Population Trends by Selected Nationalities (2010-2023)', 
                      xaxis_title='Year', yaxis_title='Population')
    st.plotly_chart(fig)

# Bar chart for total population by continent for two selected years
def plot_total_by_continent_for_two_years(year1, year2):
    year1 = int(year1)
    year2 = int(year2)
    year1_str = str(year1)
    year2_str = str(year2)

    continent_data = data.groupby('Continent').sum().iloc[:, 1:]
    year1_data = continent_data[year1_str]
    year2_data = continent_data[year2_str]

    percentage_change_values = [calculate_percentage_change(year1_data.iloc[i], year2_data.iloc[i]) for i in range(len(year1_data))]

    fig = go.Figure(data=[
        go.Bar(name=year1_str, x=continent_data.index, y=year1_data, hovertemplate='Year: ' + year1_str + '<br>Population: %{y:,}'),
        go.Bar(name=year2_str, x=continent_data.index, y=year2_data, hovertemplate='Year: ' + year2_str + '<br>Population: %{y:,}')
    ])

    for i, continent in enumerate(continent_data.index):
        fig.add_annotation(
            x=continent,
            y=max(year1_data.iloc[i], year2_data.iloc[i]) + 0.05 * max(year1_data.iloc[i], year2_data.iloc[i]),
            text=f"% Change: {percentage_change_values[i]:.2%}",
            showarrow=True,
            arrowhead=2
        )

    fig.update_layout(
        title=f'Total Population by Continent for {year1} and {year2}',
        xaxis_title='Continent',
        yaxis_title='Population',
        barmode='group'
    )

    st.plotly_chart(fig)

# Plot total by countries for two selected years
def plot_total_by_countries_for_two_years(year1, year2, countries):
    year1 = int(year1)
    year2 = int(year2)
    year1_str = str(year1)
    year2_str = str(year2)
    
    countries_data = data[data['Nationality'].isin(countries)]
    countries_population = countries_data.set_index('Nationality').iloc[:, 1:]

    year1_data = countries_population[year1_str]
    year2_data = countries_population[year2_str]

    percentage_change_values = [calculate_percentage_change(year1_data.iloc[i], year2_data.iloc[i]) for i in range(len(year1_data))]

    fig = go.Figure(data=[
        go.Bar(name=year1_str, x=countries_population.index, y=year1_data, hovertemplate='Year: ' + year1_str + '<br>Population: %{y:,}'),
        go.Bar(name=year2_str, x=countries_population.index, y=year2_data, hovertemplate='Year: ' + year2_str + '<br>Population: %{y:,}')
    ])

    for i, country in enumerate(countries_population.index):
        fig.add_annotation(
            x=country,
            y=max(year1_data.iloc[i], year2_data.iloc[i]) + 0.05 * max(year1_data.iloc[i], year2_data.iloc[i]),
            text=f"% Change: {percentage_change_values[i]:.2%}",
            showarrow=True,
            arrowhead=2
        )

    fig.update_layout(
        title=f'Total Population by Selected Countries for {year1} and {year2}',
        xaxis_title='Country',
        yaxis_title='Population',
        barmode='group'
    )

    st.plotly_chart(fig)

# Function to prepare and plot the treemap for a selected year
def plot_treemap(year):
    year_str = str(year)
    treemap_data = data[['Continent', 'Nationality', year_str]]
    
    # Remove rows with zero or negative values
    treemap_data = treemap_data[treemap_data[year_str] > 0]
    
    if treemap_data.empty:
        st.write(f"No data available for {year}")
        return
    
    fig = px.treemap(
        treemap_data, path=['Continent', 'Nationality'], values=year_str,
        title=f'Population by Continent and Nationality for {year}',
        labels={year_str: 'Population'}
    )
    fig.update_layout(
        width=800,
        height=600
    )
    fig.update_traces(hovertemplate='Year: ' + year_str + '<br>Population: %{value:,}')
    st.plotly_chart(fig)

# Prepare data for animated bubble chart
def prepare_bubble_data():
    year_columns = [col for col in data.columns if col.isdigit()]
    bubble_data = data.melt(id_vars=['Continent', 'Nationality'], value_vars=year_columns, var_name='Year', value_name='Population')
    bubble_data['Year'] = bubble_data['Year'].astype(int)
    bubble_data = bubble_data.dropna(subset=['Population'])
    bubble_data = bubble_data[bubble_data['Population'] > 0]  # Ensure only positive values
    return bubble_data

# Create animated bubble chart
def plot_bubble_chart():
    bubble_data = prepare_bubble_data()
    
    fig = px.scatter(
        bubble_data, x='Year', y='Population', size='Population', color='Continent', 
        hover_name='Nationality', animation_frame='Year', animation_group='Nationality',
        title='Population Trends Over Time by Continent',
        labels={'Population': 'Population'},
        range_x=[1995, 2023],
        range_y=[0, 200000]
    )

    fig.update_layout(
        width=900,
        height=600,
        xaxis=dict(
            tickmode='linear',
            tick0=2010,
            dtick=1
        )
    )

    fig.update_traces(
        marker=dict(opacity=0.7, sizemode='diameter'),
        selector=dict(mode='markers'),
        hovertemplate='Year: %{x}<br>Population: %{y:,}<br>Continent: %{marker.color}<br>Nationality: %{hovertext}<br>'
    )

    st.plotly_chart(fig)

# Streamlit app
st.title('Population Analysis')
st.markdown('*Note: Figures exclude Stateless and Unknown Nationalities*') 

# Plot total population trends
st.header('Total Population Trends')
plot_total_data()

# Plot population trends by continent
st.header('Population Trends by Continent')
plot_continent_data()

# Plot population trends by selected nationalities
st.header('Population Trends by Selected Nationalities')

# Nationality selection
nationalities = st.multiselect('Select Nationalities:', data['Nationality'].unique(), default=['Brazilian', 'Mexican', 'Japanese', 'Swedish'])
plot_nationality_data(nationalities)

# Treemap for a selected year
st.header('Population Treemap')
year = st.selectbox('Select Year:', [str(year) for year in range(2010, 2024)], key='year')
plot_treemap(year)

# Bar chart for total population by continent for two selected years
st.header('Total Population by Continent for Two Selected Years')

# Streamlit widgets for year selection
year1 = st.selectbox('Select Year 1:', [str(year) for year in range(2010, 2024)], key='year1_continent')
year2 = st.selectbox('Select Year 2:', [str(year) for year in range(2010, 2024)], key='year2_continent')

plot_total_by_continent_for_two_years(year1, year2)

# Plot total by countries for two selected years
st.header('Total Population by Selected Countries for Two Selected Years')

# Streamlit widgets for year selection and country selection
year1_countries = st.selectbox('Select Year 1:', [str(year) for year in range(2010, 2024)], key='year1_countries')
year2_countries = st.selectbox('Select Year 2:', [str(year) for year in range(2010, 2024)], key='year2_countries')

# Streamlit widget for countries selection
countries = st.multiselect('Select Countries:', data['Nationality'].unique(), default=['Mexican', 'Polish', 'Chilean', 'Peruvian'])

plot_total_by_countries_for_two_years(year1_countries, year2_countries, countries)

# Plot animated bubble chart
st.header('Animated Bubble Chart of Population Trends')
plot_bubble_chart()

st.markdown('Made by [Valentin Mendez](https://www.linkedin.com/in/valentemendez/) using information from [Overheid.nl](https://data.overheid.nl/dataset/268-immi--en-emigratie--per-maand--migratieachtergrond--geslacht#panel-resources)')

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

st.markdown(hide_st_style, unsafe_allow_html=True)