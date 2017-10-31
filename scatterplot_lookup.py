#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:03:14 2017

@author: Beth
"""

import pandas as pd
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import row, column, widgetbox
from bokeh.models import CustomJS, TextInput, Div
from bokeh.palettes import RdBu # , OrRd, Plasma

output_file("lookup_scatterplot.html")

# Columns: "Auth", "Works", "AvgTotal", "Avg6", "AvgLater", "Title", "titleval"
data2 = pd.read_pickle("/Users/Beth/Python/AuthorSumTotal_Agg")
# AuthorSumWorks")

data2 = data2.loc[(data2["Works"])**2 + (data2["AvgTotal"])**2 > 400]
data2 = data2.sort_values(["Avg6"])

colors = ["blue" for i in range(len(data2))]
label = ["" for i in range(len(data2))]
for i in range(len(data2)):
    trendy = data2.iloc[i]["Avg6"]
    if trendy > 20:
        colors[i] = RdBu[5][4]
        label[i] = "AMCR, new releases: 20+"
    elif trendy > 5:
        colors[i] = RdBu[5][3]
        label[i] = "AMCR, new releases: 5-20"
    elif trendy > 2:
        colors[i] = RdBu[5][2]
        label[i] = "AMCR, new releases: 2-5"
    elif trendy > 1:
        colors[i] = RdBu[5][1]
        label[i] = "AMCR, new releases: 1-2"
    else:
        colors[i] = RdBu[5][0]
        label[i] = "AMCR, new releases: <1"


sizes = [10 for i in range(len(data2))]
opacity = [1 for i in range(len(data2))]
source = ColumnDataSource(data=dict(x=data2["Works"],
                                    y=data2["AvgTotal"],
                                    auth=data2["Auth"],
                                    avg=data2["Avg6"],
                                    title=data2["Title"],
                                    tval=data2["titleval"],
                                    color=colors,
                                    size=sizes,
                                    opacity=opacity,
                                    label=label
                                    ))

# Set up the main plot
p = figure(plot_width=800, plot_height=800,
           title="")
p.yaxis[0].axis_label = 'Average monthly checkouts per author'
p.xaxis[0].axis_label = 'Number of ebooks'

callback = CustomJS(args=dict(source=source), code="""
    var text = text.value

    var data = source.data
    authors = data["auth"]
    color = data["color"]
    size = data["size"]
    opacity = data["opacity"]
    works = data["x"]
    y = data["y"]
    A6 = data["avg"]
    title = data["title"]
    tval = data["tval"]

    var found = 0
    var para = para
    var para2 = para2
    var para3 = para3
    var para4 = para4

    if(" ".indexOf(text) == -1){
        for (i=0;i<authors.length; i++){
                if(authors[i].indexOf(text) != -1){
                        console.log(i)
                        size[i]=20
                        found = 1
                        opacity[i] = 1
                        para.text= authors[i]
                        para2.text = "Number of ebooks: " + works[i]
                        para3.text = "Monthly checkouts: " + y[i].toFixed(2)
                        para4.text = "Monthly checkouts, new releases "
                            + "(first 6 months): " + A6[i].toFixed(2)
                        para5.text = "Most popular title: " + title[i]
                        }
            else{
                    size[i]=10
                    opacity[i]=0.05
            }
        }
    }

    if(found==0){
            for( i=0;i<authors.length;i++){
                    opacity[i] = 1
                    size[i]=10
            }
            para.text = ""
            para2.text = ""
            para3.text = ""
            para4.text = ""
            para5.text = ""
    }

    source.trigger("change")
""")
# source.change.emit();

auth_box = TextInput(value="Author Name", callback=callback)
callback.args["text"] = auth_box

r1 = p.circle("x", "y", size="size", fill_color="color",
              alpha="opacity", legend="label", source=source)
p.legend.location = "top_center"
p.add_tools(HoverTool(renderers=[r1], tooltips=[
    ("Author", "@auth"),
    ("Number of ebooks", "@x"),
    ("Average Checkouts, first 6 months", "@avg"),
    ("Average Checkouts, overall", "@y"),
    ]))


# Plot the extra patch, in the lower left corner
x = [0]
x.extend([i for i in range(0, 21)])
y = [0]
y.extend([float((max(390-i**2, 0))**(0.5)) for i in range(0, 21)])
r5 = p.patch(x, y, alpha=0.2, line_width=2)
# Hover tool appears to be broken for patches ?!
# p.add_tools(HoverTool(renderers=[r5], tooltips=[
#    ("Various authors", "")
#    ]))

# Set up the blocks of text (established in the custom callback)
para = Div(text="", width=200, height=20)
callback.args["para"] = para

para2 = Div(text="", width=200, height=20)
callback.args["para2"] = para2

para3 = Div(text="", width=200, height=20)
callback.args["para3"] = para3

para4 = Div(text="", width=200, height=30)
callback.args["para4"] = para4

para5 = Div(text="", width=200, height=30)
callback.args["para5"] = para5

intro = Div(text="<h2>Explore popularity by author</h2> \
            Browse authors by hovering over dots.\
            <br>Search by typing an author name into the box on the right \
            (note that searching is case sensitive). \
            <br>Reset by doing an empty search.",
            width=600)
final = Div(text="<h2>Notes:</h2>\
            The pale area in the lower left of the chart represents \
            all of the authors whose popularity, measured both by number of \
            works and checkout rate, fell below a given threshold.\
            <br><br>The average monthly checkout rate (AMCR) is the sum of \
            the AMCRs for each ebook written by that \
            author. The average for each ebook is taken only over the months \
            since it has been published. \
            <br><br>The AMCR for new releases is the \
            sum of the AMCRs for each ebook, in the \
            first 6 months after that ebook was published. This indicates how \
            trendy an author is, i.e. how much people rush out to read a new \
            release. None of the ebooks in the data set listed a publication \
            date prior to 2005.", width=400)

layout = column(
    intro,
    row(
        p,
        widgetbox(auth_box, para, para2, para3, para4, para5),
        ),
    final
    )


show(layout)
