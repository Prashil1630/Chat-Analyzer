#app.py
import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from wordcloud import WordCloud
import pandas as pd

st.set_page_config(layout="wide")  # Set the page layout to wide

st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Decode bytes data using UTF-8 and assign it to a variable
    decoded_data = bytes_data.decode("utf-8")
    # Display the decoded data
    #st.text(decoded_data)

    df = preprocessor.preprocess(decoded_data)

    #st.dataframe(df)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show Analysis w.r.t.", user_list)

    if st.sidebar.button("Show Analysis"):
        num_messages, num_words, num_media_messages, num_links = helper.fetch_stat(selected_user, df)

        #Stats Area
        st.title("TOP STATISTICS")
        cols = st.columns(4)
        with cols[0]:
            st.header("Total Messages")
            st.title(num_messages)

        with cols[1]:
            st.header("Total Words")
            st.title(num_words)

        with cols[2]:
            st.header("Media Shared")
            st.title(num_media_messages)

        with cols[3]:
            st.header("Total Links")
            st.title(num_links)

        #monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user,df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # create activity map
        st.title("Activity Map")
        cols = st.columns(2)

        with cols[0]:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values,color= 'purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with cols[1]:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color= 'orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        #heatmap
        st.title("Weekly Activity Map")
        user_heatmap= helper.user_heatmap(selected_user,df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest user in the group(at the group level)
        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x,new_df = helper.most_busy_user(df)
            fig, ax = plt.subplots()

            cols = st.columns(2)

            with cols[0]:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with cols[1]:
                st.dataframe(new_df)


        #WordCloud
        st.title("WorldCloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)



        #most Common word
        st.title("Most Common Words")
        most_common_df = helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        #st.dataframe(most_common_df)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")
        cols = st.columns(2)

        with cols[0]:
            st.dataframe(emoji_df)
        with cols[1]:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct = '%0.2f')
            st.pyplot(fig)


        #Sentiment Analysis
        st.header("Sentiment Over Time")
        daily_timeline = helper.daily_timeline(selected_user, df)
        sentiment_data = helper.sentiment_analysis(daily_timeline, df)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Use correct column names based on your DataFrame
        date_col = 'only_date'  # Update this if necessary
        if date_col not in sentiment_data.columns:
            raise KeyError(f"Column '{date_col}' not found in sentiment_data DataFrame")
        if 'positive' not in sentiment_data.columns:
            raise KeyError("Column 'positive' not found in sentiment_data DataFrame")
        if 'negative' not in sentiment_data.columns:
            raise KeyError("Column 'negative' not found in sentiment_data DataFrame")

        # Plot positive messages above the x-axis
        ax.bar(sentiment_data[date_col], sentiment_data['positive'], label='Positive', color='green')

        # Plot negative messages below the x-axis by multiplying by -1
        ax.bar(sentiment_data[date_col], -sentiment_data['negative'], label='Negative', color='red')

        # Labeling
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Messages")
        ax.legend()

        st.pyplot(fig)

        # Calculate total counts
        positive_count = sentiment_data['positive'].sum()
        negative_count = sentiment_data['negative'].sum()
        neutral_count = sentiment_data['neutral'].sum()

        # Data for pie chart
        labels = ['Positive', 'Negative', 'Neutral']
        values = [positive_count, negative_count, neutral_count]
        colors = ['green', 'red', 'rgb(131, 201, 255)']
        # Create pie chart
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3, marker=dict(colors=colors))])
        fig.update_layout(title='Sentiment Distribution')

        st.plotly_chart(fig)