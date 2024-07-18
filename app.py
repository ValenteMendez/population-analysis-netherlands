import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Migration Analysis Netherlands",
                   page_icon=":bar_chart:")

# Load the dataset
file_path = 'merged_immigration_data - Clean DB immigration.csv'
data = pd.read_csv(file_path)

# Function to filter the dataset based on the selected type of data
def filter_data_by_type(data, data_type):
    return data[data['Type of data'] == data_type]

# Helper function to get total data per year
def get_total_data_per_year(data):
    return data.iloc[:, 3:].sum()

# Helper function to get data by continent
def get_continent_data(data):
    return data.groupby('Continent').sum().iloc[:, 1:]

# Helper function to get data by selected nationalities
def get_nationality_data(data, nationalities):
    return data[data['Nationality'].isin(nationalities)]

# Total data trends over the years
def plot_total_data(data_type):
    filtered_data = filter_data_by_type(data, data_type)
    total_data_per_year = get_total_data_per_year(filtered_data)
    fig = px.line(total_data_per_year, x=total_data_per_year.index, y=total_data_per_year.values,
                  labels={'index': 'Year', 'y': f'Number of people - {data_type.lower()} '}, 
                  title=f'Migratory movements in the Netherlands (2010-2023) - {data_type}')
    st.plotly_chart(fig)

# Data trends by continent
def plot_continent_data(data_type):
    filtered_data = filter_data_by_type(data, data_type)
    continent_data = get_continent_data(filtered_data)
    fig = go.Figure()
    for continent in continent_data.index:
        fig.add_trace(go.Scatter(x=continent_data.columns, y=continent_data.loc[continent], mode='lines+markers', name=continent))
    fig.update_layout(title=f'Migratory movements trends by continent (2010-2023) - {data_type}', 
                      xaxis_title='Year', yaxis_title=f'Number of people - {data_type.lower()}')
    st.plotly_chart(fig)

# Data trends by selected nationalities
def plot_nationality_data(data_type, nationalities):
    filtered_data = filter_data_by_type(data, data_type)
    nationality_data = get_nationality_data(filtered_data, nationalities)
    fig = go.Figure()
    for nationality in nationalities:
        fig.add_trace(go.Scatter(x=nationality_data.columns[3:], 
                                 y=nationality_data[nationality_data['Nationality'] == nationality].iloc[0, 3:], 
                                 mode='lines+markers', name=nationality))
    fig.update_layout(title=f'Migratory movements trends by selected nationalities (2010-2023) - {data_type}', 
                      xaxis_title='Year', yaxis_title=f'Number of people - {data_type.lower()}')
    st.plotly_chart(fig)

# Function to prepare and plot the treemap for a selected year and data type
def plot_treemap(year, data_type):
    filtered_data = filter_data_by_type(data, data_type)
    year_str = str(year)
    treemap_data = filtered_data[['Continent', 'Nationality', year_str]]
    
    # Remove rows with zero or negative values
    treemap_data = treemap_data[treemap_data[year_str] > 0]
    
    if treemap_data.empty:
        st.write(f"No data available for {data_type} in {year}")
        return
    
    fig = px.treemap(
        treemap_data, path=['Continent', 'Nationality'], values=year_str,
        title=f'Migration movements by continent and nationality for {year} - {data_type}',
        labels={year_str: 'Number of migration movements'}
    )
    fig.update_layout(
        width=800,
        height=600
    )
    st.plotly_chart(fig)

# Calculate percentage change
def calculate_percentage_change(start_value, end_value):
    if start_value == 0:
        return float('inf')
    return (end_value - start_value) / start_value

# Bar chart for total migration movements by continent for two selected years
def plot_total_by_continent_for_two_years(year1, year2, data_type):
    year1 = int(year1)
    year2 = int(year2)
    year1_str = str(year1)
    year2_str = str(year2)

    filtered_data = filter_data_by_type(data, data_type)
    continent_data = filtered_data.groupby('Continent').sum().iloc[:, 1:]
    year1_data = continent_data[year1_str]
    year2_data = continent_data[year2_str]

    percentage_change_values = [calculate_percentage_change(year1_data.iloc[i], year2_data.iloc[i]) for i in range(len(year1_data))]

    fig = go.Figure(data=[
        go.Bar(name=year1_str, x=continent_data.index, y=year1_data),
        go.Bar(name=year2_str, x=continent_data.index, y=year2_data)
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
        title=f'Total migration movements by continent for {year1} and {year2} - {data_type}',
        xaxis_title='Continent',
        yaxis_title='Number of migration movements',
        barmode='group'
    )

    st.plotly_chart(fig)

# Bar chart for total migration movements by selected countries for two selected years
def plot_total_by_countries_for_two_years(year1, year2, countries, data_type):
    year1 = int(year1)
    year2 = int(year2)
    year1_str = str(year1)
    year2_str = str(year2)
    
    filtered_data = filter_data_by_type(data, data_type)
    countries_data = filtered_data[filtered_data['Nationality'].isin(countries)]
    countries_migration = countries_data.set_index('Nationality').iloc[:, 1:]

    year1_data = countries_migration[year1_str]
    year2_data = countries_migration[year2_str]

    percentage_change_values = [calculate_percentage_change(year1_data.iloc[i], year2_data.iloc[i]) for i in range(len(year1_data))]

    fig = go.Figure(data=[
        go.Bar(name=year1_str, x=countries_migration.index, y=year1_data),
        go.Bar(name=year2_str, x=countries_migration.index, y=year2_data)
    ])

    for i, country in enumerate(countries_migration.index):
        fig.add_annotation(
            x=country,
            y=max(year1_data.iloc[i], year2_data.iloc[i]) + 0.05 * max(year1_data.iloc[i], year2_data.iloc[i]),
            text=f"% Change: {percentage_change_values[i]:.2%}",
            showarrow=True,
            arrowhead=2
        )

    fig.update_layout(
        title=f'Total migration movements by selected countries for {year1} and {year2} - {data_type}',
        xaxis_title='Country',
        yaxis_title='Number of migration movements',
        barmode='group'
    )

    st.plotly_chart(fig)

# Prepare data for animated bubble chart
def prepare_bubble_data(data_type):
    filtered_data = filter_data_by_type(data, data_type)
    year_columns = [col for col in filtered_data.columns if col.isdigit()]
    bubble_data = filtered_data.melt(id_vars=['Continent', 'Nationality'], value_vars=year_columns, var_name='Year', value_name='Migration Movements')
    bubble_data['Year'] = bubble_data['Year'].astype(int)
    bubble_data = bubble_data.dropna(subset=['Migration Movements'])
    bubble_data = bubble_data[bubble_data['Migration Movements'] > 0]  # Ensure only positive values
    return bubble_data

# Create animated bubble chart
def plot_bubble_chart(data_type):
    bubble_data = prepare_bubble_data(data_type)
    
    fig = px.scatter(
        bubble_data, x='Year', y='Migration Movements', size='Migration Movements', color='Continent', 
        hover_name='Nationality', animation_frame='Year', animation_group='Nationality',
        title=f'Migration movements trends over time by continent - {data_type}',
        labels={'Migration Movements': 'Number of migration movements'},
        range_x=[1995, 2023],
        range_y=[0, 30000]
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
        selector=dict(mode='markers')
    )

    st.plotly_chart(fig)

# Function to calculate the initial population as the balance value of 1995
def calculate_initial_population(data, continent, country):
    continent_data = data[data['Continent'] == continent]
    country_data = continent_data[continent_data['Nationality'] == country]
    balance_1995 = country_data[(country_data['Type of data'] == 'Balance')]['1995']
    if not balance_1995.empty:
        return balance_1995.values[0]
    else:
        return 0

# Function to calculate total population for the selected country and years
def calculate_total_population(data, continent, country, start_year, end_year, initial_population):
    # Filter data for the given continent and country
    continent_data = data[data['Continent'] == continent]
    country_data = continent_data[continent_data['Nationality'] == country]
    
    # Convert start_year and end_year to string
    start_year = str(start_year)
    end_year = str(end_year)
    
    # Filter columns by the selected years
    year_columns = [col for col in data.columns if col.isdigit() and start_year <= col <= end_year]
    
    # Separate immigration and emigration data
    immigration_data = country_data[country_data['Type of data'] == 'Immigration'][year_columns].sum(axis=1).values
    emigration_data = country_data[country_data['Type of data'] == 'Emigration'][year_columns].sum(axis=1).values

    # Handle cases where there might be no data
    if immigration_data.size == 0:
        immigration_data = [0]
    if emigration_data.size == 0:
        emigration_data = [0]

    # Calculate the balance
    balance = immigration_data[0] - emigration_data[0]

    # Calculate total population
    total_population = initial_population + balance
    return total_population

# Streamlit app
st.title('Migratory Movements Analysis')
st.markdown('*Note: Figures exclude Migration background = Dutch background*') 

# Data type selection
data_type = st.selectbox('Select Data Type:', ['Immigration', 'Emigration', 'Balance'], key='data_type_1')

# Plot total data trends
st.header('Total Data Trends')
plot_total_data(data_type)

# Plot data trends by continent
st.header('Data Trends by Continent')
plot_continent_data(data_type)

# Plot data trends by selected nationalities
st.header('Data Trends by Selected Nationalities')

# Nationality selection
nationalities = st.multiselect('Select Nationalities:', data['Nationality'].unique(), default=['Brazil', 'Mexico', 'Japan', 'Zweden'])


plot_nationality_data(data_type, nationalities)

# Function to get data by continent
def get_continent_data(data, continent):
    return data[data['Continent'] == continent]

# Data trends by continent and country
def plot_continent_and_country_data(data_type, continent):
    filtered_data = filter_data_by_type(data, data_type)
    continent_data = get_continent_data(filtered_data, continent)
    
    fig = go.Figure()
    for country in continent_data['Nationality'].unique():
        country_data = continent_data[continent_data['Nationality'] == country].iloc[:, 3:].T
        country_data.columns = [country]
        fig.add_trace(go.Scatter(
            x=country_data.index, 
            y=country_data[country], 
            mode='lines+markers', 
            name=country,
            hovertemplate='<b>Country:</b> %{text}<br><b>Year:</b> %{x}<br><b>Value:</b> %{y}',
            text=[country] * len(country_data)
        ))
    
    fig.update_layout(
        title=f'Migratory movements trends for {continent} (2010-2023) - {data_type}', 
        xaxis_title='Year', 
        yaxis_title=f'Number of people - {data_type.lower()}',
        hovermode='closest'
    )
    st.plotly_chart(fig)

# Streamlit app for continent and country trends
st.title('Migratory Movements Analysis by Continent')

# Interactive dropdown for data type selection
data_type = st.selectbox('Select Data Type:', ['Immigration', 'Emigration', 'Balance'], key='data_type_2')

# Interactive dropdown for continent selection
continent = st.selectbox('Select Continent:', data['Continent'].unique(), key='continent')

# Plot data trends by continent and country
plot_continent_and_country_data(data_type, continent)

# Treemap for a selected year and data type
st.header('Treemap of Migration Movements')
year = st.selectbox('Select Year:', [str(year) for year in range(1995, 2023)], key='year')
data_type_treemap = st.selectbox('Select Data Type for Treemap:', ['Immigration', 'Emigration', 'Balance'], key='data_type_3')
plot_treemap(year, data_type_treemap)

# Streamlit app for total population calculation
st.title('Total Population Calculation')
st.markdown('*Note: Suggested start year to calculate terminal population is 1996.*') 

# Interactive widgets for total population calculation
continent_pop = st.selectbox('Select Continent:', data['Continent'].unique(), key='continent_dropdown')

# Update countries based on selected continent
countries = data[data['Continent'] == continent_pop]['Nationality'].unique()
country_pop = st.selectbox('Select Country:', countries, key='country_dropdown')

start_year = st.selectbox('Select Start Year:', [str(year) for year in range(1996, 2023)], key='start_year_dropdown')
end_year = st.selectbox('Select End Year:', [str(year) for year in range(1996, 2023)], index=len(range(1996, 2023))-1, key='end_year_dropdown')

# Calculate initial population
initial_population = calculate_initial_population(data, continent_pop, country_pop)
st.write(f'Initial Population (Balance 1995): {initial_population:,}')

# Calculate total population on button click
if st.button('Calculate Total Population'):
    total_population = calculate_total_population(data, continent_pop, country_pop, start_year, end_year, initial_population)
    st.write(f'Total population of {country_pop} including initial population: {total_population:,.0f}')

# Plot total by continent for two selected years
st.header('Total Migration Movements by Continent for Two Selected Years')

# Streamlit widgets for year selection
year1 = st.selectbox('Select Year 1:', [str(year) for year in range(2010, 2023)], key='year1_continent')
year2 = st.selectbox('Select Year 2:', [str(year) for year in range(2010, 2023)], key='year2_continent')

# Streamlit widget for data type selection
data_type_continent = st.selectbox('Select Data Type:', ['Immigration', 'Emigration', 'Balance'], key='data_type_continent')

plot_total_by_continent_for_two_years(year1, year2, data_type_continent)

# Plot total by countries for two selected years
st.header('Total Migration Movements by Selected Countries for Two Selected Years')

# Streamlit widgets for year selection and country selection
year1_countries = st.selectbox('Select Year 1:', [str(year) for year in range(2010, 2023)], key='year1_countries')
year2_countries = st.selectbox('Select Year 2:', [str(year) for year in range(2010, 2023)], key='year2_countries')

# Streamlit widget for countries selection
countries = st.multiselect('Select Countries:', data['Nationality'].unique(), default=['Mexico', 'Poland', 'Chili', 'Peru'])

# Streamlit widget for data type selection
data_type_countries = st.selectbox('Select Data Type:', ['Immigration', 'Emigration', 'Balance'], key='data_type_countries')

plot_total_by_countries_for_two_years(year1_countries, year2_countries, countries, data_type_countries)

# Plot animated bubble chart
st.header('Animated Bubble Chart of Migration Movements')

# Streamlit widget for data type selection
data_type_bubble = st.selectbox('Select Data Type for Bubble Chart:', ['Immigration', 'Emigration', 'Balance'], key='data_type_bubble')

plot_bubble_chart(data_type_bubble)

st.markdown('Made by [Valentin Mendez](https://www.linkedin.com/in/valentemendez/) using information from [Overheid.nl](https://data.overheid.nl/dataset/268-immi--en-emigratie--per-maand--migratieachtergrond--geslacht#panel-resources)')

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
   
    </style>
    """

st.markdown(hide_st_style, unsafe_allow_html=True)