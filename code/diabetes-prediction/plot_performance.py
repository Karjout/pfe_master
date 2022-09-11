import plotly.express as px
# storing plot data
lst = []


def plot_line(df, y, name, color_):
    fig = px.line(df, x="epoch", y=y)
    lst.append(fig.update_traces(
        name=name, showlegend=True, line=dict(color=color_)).data)
